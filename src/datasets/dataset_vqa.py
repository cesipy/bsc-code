from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

from .dataset_utils import BaseImageProcessor, PreTrainedTokenizerFast, torch; from . import dataset_utils

import easy_vqa

class VqaDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        image_processor: BaseImageProcessor,
        is_train: bool=True,
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor

        if is_train:
            questions, answers, image_ids = easy_vqa.get_train_questions()
            img_paths = easy_vqa.get_train_image_paths()
        else:
            questions, answers, image_ids = easy_vqa.get_test_questions()
            img_paths = easy_vqa.get_test_image_paths()


        assert len(questions) == len(answers) == len(image_ids)
        self.data = list(zip(questions, answers, image_ids))
        self.img_paths = img_paths


        # 13 possible
        all_answers = easy_vqa.get_answers()
        self.answer_vocab = {ans: idx for idx, ans in enumerate(all_answers)}
        self.num_classes = len(all_answers)

        print(f"\nAll possible answers ({len(all_answers)}): {all_answers}")
        print(f"vocab answ: {self.answer_vocab}\n")
        test_label = self.answer_vocab[answers[15]]
        print(f"\nAnswer '{answers[15]}' maps to index: {test_label}")


    def __len__(self,):
        return len(self.data)

    def __getitem__(self, index):
        """Returns dictionary matching your HM_Dataset format:
        {
            "img": img_tensor,
            "label": label_tensor,
            "text": text_tensor
        }
        """
        curr_q, curr_a, curr_img_id = self.data[index]
        curr_img_path = self.img_paths[curr_img_id]

        upscale_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # upscale
        ])


        img_embeddings = dataset_utils.process_image(
            img=curr_img_path,
            transform=upscale_transform
        )

        question_embeddings = dataset_utils.get_text_embedding(
            curr_q,
            tokenizer=self.tokenizer
        )

        # convert answer to class index
        label_tensor = torch.tensor(
            self.answer_vocab[curr_a],
            dtype=torch.long
        )

        return {
            "img": img_embeddings,      # {"pixel_values": tensor}
            "label": label_tensor,      # class index (0-12)
            "text": question_embeddings # {"input_ids": ..., "attention_mask": ...}
        }

if __name__ == "__main__":
    from transformers import ViTImageProcessor, BertTokenizerFast

    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


    dataset = VqaDataset(
        tokenizer=tokenizer,
        image_processor=image_processor,
        is_train=True
    )


    sample = dataset[0]





    dataloader= DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(dataloader))
    print("Batch image shape:", batch["img"]["pixel_values"].shape)  # [4, 1, 3, 224, 224]