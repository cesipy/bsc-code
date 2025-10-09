import torch; from torch import nn
from transformers import (
    #BERT stuff
    BertModel,
    BertConfig,
    BertTokenizer,
    BertTokenizerFast,

    # ViT stuff
    ViTConfig,
    ViTModel,
    ViTImageProcessor,

    # type hinting stuff
    PreTrainedTokenizerFast,
)

import timm

from PIL import Image

from attention import(
    Attention_Block,
    CrossAttention, CrossAttentionBlock, FeedForward_Block,
    DualAttention_Block
)


from datasets import MM_IMDB_Dataset; import datasets


from config import *
from logger import Logger

from abc import ABC, abstractmethod
class Model(nn.Module, ABC):

    def __init__(self, config):
        super(Model, self).__init__()

        self.config = config
        # loads pretrained transformers, no head for task. with transformers.BertFor.... I
        # could download pretrained transformers for specific tasks
        self.bert = BertModel.from_pretrained("google-bert/bert-base-uncased")

        # apparently transformers vit implementatins is flawed.
        # self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.vit = timm.create_model(
            VIT_MODEL_NAME,
            pretrained=True,
            num_classes=0,       # num_classes=0 removes head
            global_pool="",      # we need whole sequence for mim
        )

        # pretrain heads
        self.alignment_fc = nn.Linear(2*self.config.embedding_dim , 1)
        self.mlm = nn.Linear(self.config.embedding_dim, self.bert.config.vocab_size)    #30522

        # for hateful memes
        self.fc = nn.Sequential(
            nn.Linear(self.config.embedding_dim, FC_HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout_prob),
            nn.Linear(FC_HIDDEN_DIM, 1),
        )

        # TODO: unify, here i do multiplication in the other not
        # for mmimdb
        self.fc_imdb = nn.Sequential(
            nn.Linear(self.config.embedding_dim, FC_HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout_prob),
            nn.Linear(FC_HIDDEN_DIM, MM_IMDB_NUM_GENRES),
        )
        self.fc_vqa = nn.Sequential(
            nn.Linear(self.config.embedding_dim, FC_HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout_prob),
            nn.Linear(FC_HIDDEN_DIM, EASY_VQA_NUM_CLASSES),
        )




    @abstractmethod
    def forward(self, *args, **kwargs):
        pass



class VisionEmbeddings(nn.Module):
    """Process ViT outputs into ViLBERT format"""
    def __init__(self,):
        super().__init__()
        # ViT already gives us 768-dim embeddings, just add normalization
        self.LayerNorm = nn.LayerNorm(768, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, vit_features):
        # vit_features: [batch, 197, 768] from ViT
        # same as in vilbert
        # layernorm might break pretrained weights
        embeddings = vit_features
        # embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ViLBERT(nn.Module):

    def __init__(self, config):
        super(ViLBERT, self).__init__()

        self.bert = BertModel.from_pretrained("google-bert/bert-base-uncased")
        # apparently transformers vit implementatins is flawed.
        # self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.vit = timm.create_model(
            VIT_MODEL_NAME,
            pretrained=True,
            num_classes=0,       # num_classes=0 removes head
            global_pool="",      # we need whole sequence for mim
        )
        self.bert_layers = self.bert.encoder.layer
        self.bert_embeddings = self.bert.embeddings


        self.vision_embeddings = VisionEmbeddings()


        # bad naming, copied from og-vilbert
        # self.v_biattention_ids = [4, 7, 10, 11]
        # self.t_biattention_ids = [6, 8, 10, 11]
        self.v_biattention_ids = config.vision_cross_attention_layers
        self.t_biattention_ids = config.text_cross_attention_layers
        assert len(self.v_biattention_ids) == len(self.t_biattention_ids)

        # for freezing, TODO
        self.fixed_t_layer = 3
        self.fixed_v_layer = 3
        self.depth = config.depth       # TODO: problem here
        self.depth = 12
        # there are the original 12 layers per encoder
        # self.depth = config.depth + len(self.v_biattention_ids)

        self.config = config
        self.c_layers = nn.ModuleList()
        for i in range(len(self.v_biattention_ids)):
            self.c_layers.append(CrossAttentionBlock(
                dim=EMBEDDING_DIM,
                heads=NUM_BI_ATTENTION_HEADS,
                dropout=config.dropout_prob,
            ))

        # pretrain heads
        self.alignment_fc = nn.Linear(self.config.embedding_dim , 1)
        self.mlm = nn.Linear(self.config.embedding_dim, self.bert.config.vocab_size)    #30522

        if CLS_FUSION_METHOD == "concat":
            input_dim = self.config.embedding_dim * 2
        else:
            input_dim = self.config.embedding_dim
        # for hateful memes
        self.fc = nn.Sequential(
            nn.Linear(input_dim, FC_HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout_prob),
            nn.Linear(FC_HIDDEN_DIM, 1),
        )
        # TODO: unify, here i do multiplication in the other not
        # for mmimdb
        self.fc_imdb = nn.Sequential(
            nn.Linear(input_dim, FC_HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout_prob),
            nn.Linear(FC_HIDDEN_DIM, MM_IMDB_NUM_GENRES),
        )
        self.fc_vqa = nn.Sequential(
            nn.Linear(input_dim, FC_HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout_prob),
            nn.Linear(FC_HIDDEN_DIM, EASY_VQA_NUM_CLASSES),
        )
        self.fc_upmc = nn.Sequential(
            nn.Linear(input_dim, FC_HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout_prob),
            nn.Linear(FC_HIDDEN_DIM, UPMC_NUM_CLASSES),
        )




    def get_extended_attention_mask(self, attention_mask: torch.Tensor, dtype=None):
        if attention_mask is not None:
            attention_mask = attention_mask.float()
            extended_attention_mask = attention_mask[:, None, None, :]
            #extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None
        return extended_attention_mask

    def forward_coattention(
        self,
        text_input_ids,
        text_attention_mask=None,
        text_token_type_ids=None,
        image_pixel_values=None,
        image_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        extract_cls=True ,          # used for pretraining. should the cls token be extracted?
                                    # not extracted for pretraining, extracted for downstream tasks
        save_intermediate_representations=False
    ):

        # TODO: rethink
        # if not extract_cls:
        #     assert save_intermediate_representations == False, "not working"
        extended_attention_mask = self.get_extended_attention_mask(
            text_attention_mask,
            dtype=next(self.bert.parameters()).dtype
        )

        text_embedding = self.bert_embeddings(
            input_ids=text_input_ids,
            token_type_ids=text_token_type_ids,
        )

        # #this is wrong, according to the sourcecode, this simply skips the head
        # and skips the pooling stage in the transformer:
        # https://github.com/huggingface/pytorch-image-models/blob/0645384b3a68d0ddf4657400125bb2c68c42bc60/timm/models/vision_transformer.py#L935
        # vit_outputs = self.vit.forward_features(
        #     image_pixel_values,
        # )

        #applies conv2d to image with kernel_sz=16 and stride = 16
        #=> 14x14 patches = 196
        #https://huggingface.co/spaces/Roll20/pet_score/blame/main/lib/timm/models/layers/patch_embed.py
        x = self.vit.patch_embed(image_pixel_values)    #[bs, 196, dim]
        cls = self.vit.cls_token.expand(x.shape[0], -1, -1)  # [bs, 1, dim] stole cls token impl from timm
        x = torch.cat((cls, x), dim=1)  # [bs, 197, dim]

        # apply positional dropout + positional embedding
        x = self.vit.pos_drop(x + self.vit.pos_embed)

        # no need to droput anymore
        #vision_embeddings = self.vision_embeddings(x)
        vision_embeddings = x


        v_start = 0
        t_start = 0
        count   = 0

        intermediate_representations = [] if save_intermediate_representations else None
        tmp_t = []
        tmp_v = []
        for v_layer_id, t_layer_id in zip(self.v_biattention_ids, self.t_biattention_ids):
            # needed for freezing layers
            # v_end = v_layer_id
            # t_end = t_layer_id
            # print(f"v_start: {v_start}, v_end: {v_end}, t_start: {t_start}, t_end: {t_end}")
            # # for idx in range(t_start, t_end)
            # for idx in range (t_start, self.fixed_t_layer):
            #     with torch.no_grad():
            #         text_embedding = self.bert_layers[idx](
            #             text_embedding,
            #             attention_mask=text_attention_mask,
            #         )[0]
            #         t_start = self.fixed_t_layer
            #         print(f"Text only layer {idx} done")

            for i in range(t_start, t_layer_id):
                text_embedding = self.bert_layers[i](
                    text_embedding,
                    attention_mask=extended_attention_mask,
                )[0]
                if save_intermediate_representations:
                    dict_entry = {
                        "layer": f"t{i}",
                        "text_embedding": text_embedding.clone(),
                        "is_cross_attention": False
                    }
                    tmp_t.append(dict_entry)


            for i in range(v_start, v_layer_id):
                vision_embeddings = self.vit.blocks[i](
                    vision_embeddings,
                )
                if save_intermediate_representations:
                    dict_entry = {
                        "layer": f"v{i}",
                        "vision_embedding": vision_embeddings.clone(),
                        "is_cross_attention": False
                    }
                    tmp_v.append(dict_entry)



            text_embedding, vision_embeddings = self.c_layers[count](
                text_embedding,
                vision_embeddings,
                text_mask=extended_attention_mask,
            )
            ## skip coats for now
            # dict_entry_t = {
            #     "layer": f"c{count}",
            #     "text_embedding": text_embedding.clone(),
            #     "is_cross_attention": True
            # }
            # dict_entry_v = {
            #     "layer": f"c{count}",
            #     "vision_embedding": vision_embeddings.clone(),
            #     "is_cross_attention": True
            # }
            # if save_intermediate_representations:
            #     tmp_t.append(dict_entry_t)
            #     tmp_v.append(dict_entry_v)


            t_start = t_layer_id
            v_start = v_layer_id
            count  += 1

        # remains
        for i in range(t_start, len(self.bert_layers)):
            text_embedding = self.bert_layers[i](text_embedding, attention_mask=extended_attention_mask)[0]
            if save_intermediate_representations:
                dict_entry = {
                    "layer": f"t{i}",
                    "text_embedding": text_embedding.clone(),
                    "is_cross_attention": False
                }
                tmp_t.append(dict_entry)

        for i in range(v_start, len(self.vit.blocks)):
            vision_embeddings = self.vit.blocks[i](vision_embeddings)
            if save_intermediate_representations:
                dict_entry = {
                    "layer": f"v{i}",
                    "vision_embedding": vision_embeddings.clone(),
                    "is_cross_attention": False
                }
                tmp_v.append(dict_entry)


        if extract_cls:
            text_embedding = text_embedding[:, 0, :]      # [batch_size, hidden_dim]
            vision_embeddings = vision_embeddings[:, 0, :]  # [batch_size, hidden_dim]

        # print(len(tmp_t), len(tmp_v))
        # currently only the 12 layers in analysis

        if save_intermediate_representations:
            for i, tup in enumerate(zip(tmp_t, tmp_v)):
                dict_entry = {
                    "layer": i,
                    "text_embedding": tup[0]["text_embedding"],
                    "vision_embedding": tup[1]["vision_embedding"],
                    "is_cross_attention": i in self.v_biattention_ids or i in self.t_biattention_ids
                }
                intermediate_representations.append(dict_entry)
            return text_embedding, vision_embeddings, intermediate_representations
        return text_embedding, vision_embeddings


    def forward(self,
        text_input_ids,
        text_attention_mask=None,
        text_token_type_ids=None,
        image_pixel_values=None,
        image_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        save_intermediate_representations=False,
    ):
        #TODO: debugging
        # save_intermediate_representations= True

        if save_intermediate_representations:
            text_embedding, image_embedding, save_intermediate_representations = self.forward_coattention(
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                text_token_type_ids=text_token_type_ids,
                image_pixel_values=image_pixel_values,
                image_attention_mask=image_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                extract_cls=True,       #?
                save_intermediate_representations=True
            )
            return text_embedding, image_embedding, save_intermediate_representations

        text_embedding, image_embedding = self.forward_coattention(
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            text_token_type_ids=text_token_type_ids,
            image_pixel_values=image_pixel_values,
            image_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            extract_cls=True,
            save_intermediate_representations=False
        )

        return text_embedding, image_embedding


    def forward_pretrain(
        self,
        text_input_ids,
        text_attention_mask=None,
        text_token_type_ids=None,
        image_pixel_values=None,
        image_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        tasks:list[str]= None,      #TODO: make tasks an enunm
    ):

        assert len(tasks) == 1, "only one task at a time"

        if "mlm" in tasks:
            text_full_seqs, img_full_seqs = self.forward_coattention(
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                text_token_type_ids=text_token_type_ids,
                image_pixel_values=image_pixel_values,
                image_attention_mask=image_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                extract_cls=False,
                save_intermediate_representations=False
            )

            return text_full_seqs, img_full_seqs

        if "mim" in tasks:
            text_seqs, vision_seqs = self.forward_coattention(
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                text_token_type_ids=text_token_type_ids,
                image_pixel_values=image_pixel_values,
                image_attention_mask=image_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                extract_cls=False
            )

            return text_seqs, vision_seqs

        if "alignment_prediction" in tasks:
            text_embedding, image_embedding = self.forward_coattention(
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                text_token_type_ids=text_token_type_ids,
                image_pixel_values=image_pixel_values,
                image_attention_mask=image_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                extract_cls=True, # only cls needed. nothing else
            )
            return text_embedding, image_embedding

    def save_model(self, save_path):
        """Save model state dict and config"""
        if hasattr(self, "_orig_mod"):
            # even if model is compiled, the orig_mod is also updated when training
            # compiled model cannot be loaded.
            model_state_dict = self._orig_mod.state_dict()
        else:
            model_state_dict = self.state_dict()

        checkpoint = {
            "model_state_dict": model_state_dict,
            "config": self.config.__dict__,
        }
        torch.save(checkpoint, save_path)
        print(f"model saved to {save_path}")

    @classmethod
    def load_model(cls, load_path, device='cpu'):
        """Load model from saved checkpoint"""
        checkpoint = torch.load(load_path, map_location=device, weights_only=False)
        config = ViLBERTConfig()
        config.__dict__.update(checkpoint['config'])
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])

        print(f"model loaded from {load_path}")
        return model





if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ViLBERT().to(device)
    print(f"Model loaded successfully")

    dataloader, _ = datasets.get_mmimdb_datasets(
        train_test_ratio=0.8,
        batch_size=1,
        num_workers=0,
        prefetch_factor=None
    )

    for data_dict in dataloader:
        label = data_dict["label"].to(device)
        # its not possible to send dicts to device, so do it for every value in dict.
        text = {k: v.squeeze(1).to(device) for k, v in data_dict["text"].items()}
        image = {k: v.squeeze(1).to(device) for k, v in data_dict["img"].items()}

        model(
            text_input_ids= text["input_ids"],
            text_attention_mask= text["attention_mask"],
            text_token_type_ids= text.get("token_type_ids", None),
            image_pixel_values= image["pixel_values"],
            image_attention_mask= image.get("attention_mask", None),
        )

        break


