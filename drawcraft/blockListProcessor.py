from pathlib import Path
from skimage import filters
import numpy as np
from PIL import Image
import json
import logging
import zipfile
import tools
from typing import Tuple, Dict

class BlocklistProcessor(tools.Tools):
    def __init__(self, Block_Manager) -> None:
        self.Block_Manager = Block_Manager

        self.workingPath: Path = Path(__file__).parent.resolve() / "custom_resourcepacks"
        self.resource_name: str = Block_Manager.get_resource_name()
        self.resourcePath: Path = self.workingPath.resolve() / (self.resource_name + ".zip")
        self.block_data = {}
        self.block_images_path:Path = self.Block_Manager.get_block_images_path()
        self.default_block_data: Dict = self.Block_Manager.get_default_blocks_data()

    def unpack_zip(self) -> None:
        extract_path = self.workingPath / self.resource_name
        try:
            with zipfile.ZipFile(self.resourcePath, 'r') as zip_ref:
                if extract_path.exists():
                    self.delete_folder(extract_path)
                zip_ref.extractall(extract_path) 
                logging.info(f"Pliki zostały rozpakowane do folderu: {extract_path}")
        except (zipfile.BadZipFile, FileNotFoundError, Exception) as e:
            logging.error(f"Wystąpił błąd podczas wypakowywania ZIP: {e}") 

    def conf(self, img_path: Path) -> Tuple[float, float, float]:
        img_d = Image.open(img_path)
        return self.average_image_color(img_d)

    def average_image_color(self, im: Image.Image) -> Tuple:
        if im.mode != "RGB":
            im = im.convert("RGB")
        pixels = np.array(im).reshape(-1, 3)
        return tuple(np.mean(pixels, axis=0))

    def create_lists(self) -> None:
        logging.info("Tworzenie nowej listy bloków...")
        blockPath = self.workingPath / self.resource_name / "assets" / "minecraft" / "textures" / "block"
        block_files = list(blockPath.iterdir())
        total_blocks = len(block_files)
        for i, block in enumerate(block_files):
            if block.name not in self.default_block_data:
                continue
            self.block_data[block.name] = {}
            self.block_data[block.name]["face"] = self.default_block_data[block.name]["face"]
            self.block_data[block.name]["smoothness"] = self.get_block_smoothness(block)
            self.block_data[block.name]["filter_mode"] = "normal"
            self.block_data[block.name]["colors"] = self.conf(block)
            
            if (i + 1) % 100 == 0:
                progress = (i + 1) / total_blocks * 100
                logging.info(f"Postęp: {progress:.2f}%")

        self.save_json_file(self.workingPath / f"{self.resource_name}_data.json", self.block_data)
    def save_json_file(self, save_path: Path, content):
        with open(save_path, "w") as fl:
            json.dump(content, fl, indent=4)

    def check_if_exists(self) -> bool:

        exists = (self.workingPath / f"{self.resource_name}_data.json").exists()
        logging.info("Lista bloków {}.".format("już istnieje" if exists else "nie istnieje"))
        return exists
    
    def get_block_smoothness(self, block_path: Path):
        image = Image.open(block_path)
        grayscale = np.array(image.convert('L'))
        edges = filters.sobel(grayscale)
        smoothness = (1-edges.mean())*100
        return smoothness

    def start_blp(self):
        if not self.check_if_exists():
            self.unpack_zip()
            self.create_lists()
        return self.block_data
