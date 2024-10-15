import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
from sklearn.neighbors import KDTree
# import schem
import json
from pathlib import Path
import timeit
import logging
import blockListProcessor
from typing import List, Tuple, Dict, Union
import threading


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)

#add chcecking in aplication for valid data type int, float. Add translator

class BlockManager():
    def __init__(self, state_manager):
        self.state_manager = state_manager
        self.running_path: Path = Path(__file__).parent.resolve()
        self.default_data_file_path: Path = self.running_path / "blocks_data.json"
        self.block_images_path: Path = self.running_path / "blocks"
        self.data_file_path: Path = self.default_data_file_path
        self.block_data: dict = {}
        self.nb_blocks: int = 0
        self.filter_mode_list: list[list[str]] = [[], [], []]
        self.resource_name = ""
        self.nb_blocks = 0
        self.listed_blocks = {}
        
    def get_running_path(self) -> Path:
        return self.running_path
    
    def update_data_path(self, data_path: Path, block_images_path: Path) -> None:
        self.data_file_path = data_path
        self.block_images_path = block_images_path

    def get_resource_name(self) -> str:
        return self.resource_name

    def set_resourcepack(self, resource_path: Path) -> None:
        self.resource_name = resource_path.stem
        self.data_file_path = self.default_data_file_path.parent / "custom_resourcepacks" /(self.resource_name + "_data.json")
        self.block_images_path = self.default_data_file_path.parent / "custom_resourcepacks " / self.resource_name /"assets/minecraft/textures/block"
        BlockProcessor = blockListProcessor.BlocklistProcessor(self)
        self.block_data = BlockProcessor.start_blp()
        if not self.block_data:
            self.block_data = self.load_blocks_data(self.data_file_path)
        self.nb_blocks = len(self.block_data)

    def reset_resourcepack(self) -> None:
        self.resource_name = ""
        self.data_file_path = self.default_data_file_path
        self.block_images_path = self.running_path / "blocks"
        self.block_data = self.load_blocks_data(self.default_data_file_path)
        self.nb_blocks = len(self.block_data)
    
    def create_block_list(self):
        '''
        TODO:
        - add smootheness_mode parameter to choose between Sobel and Canny.
        '''

        mode = self.state_manager.get_filter_face_mode()
        smoothness = self.state_manager.smoothness_value.get()

        logging.info(f"Ładowanie listy bloków w trybie: {mode if mode else "Full"}")

        tryb = 1

        raw_blocks = self.get_block_data().copy()
        listed_blocks = {}

        if not mode:
            mode = ['top', 'side', 'bottom', 'solid']

        if smoothness:
            sm_listed_blocks = self.prepare_smoothness(self.get_smoothness_list(), smoothness)

        for block in raw_blocks.copy().keys():
            if (tryb == 1 and block in self.filter_mode_list[1]) or (tryb == 2 and block not in self.filter_mode_list[2]):
                del raw_blocks[block]
                continue
            if self.block_data[block]["face"] not in mode:
                del raw_blocks[block]
                continue
            if smoothness:
                if any(x in block for x in sm_listed_blocks):
                    del raw_blocks[block]
                    continue
            listed_blocks[block] = raw_blocks[block]["colors"]

        logging.info(f"Lista bloków została załadowana. Zostanie zużytych: {len(listed_blocks)}/{self.get_nb_blocks()} bloków")
        self.listed_blocks = listed_blocks
    
    def prepare_smoothness(self, smooth_values: Dict, smoothness: float) -> List[str]:
        logging.info(f"Przygotowywanie listy bloków z progami gładkości: {smoothness}")

        if min(smooth_values.values()) > smoothness or max(smooth_values.values()) < smoothness:
            logging.warning(
                f"Ustawiony próg gładkości jest zbyt {('niski' if min(smooth_values.values()) > smoothness else 'wysoki')}. "
                f"Wszystkie bloki zostaną użyte. Dostępne poziomy są w przedziale: {min(smooth_values.values())} -> {max(smooth_values.values())}"
            )
            return []

        sm_listed_blocks = [block for block, value in smooth_values.items() if value <= smoothness]

        return sm_listed_blocks

    def get_listed_blocks(self):
        return self.listed_blocks
    
    def update_nb_blocks(self) -> None:
        self.nb_blocks = len(self.block_data)

    def get_data_path(self) -> Path:
        return self.data_file_path
    
    def get_block_images_path(self) -> Path:
        return self.block_images_path
    
    def get_default_block_images_path(self) -> Path:
        return self.running_path / "blocks"
    
    def load_blocks_data(self, data_path:Path = Path(__file__).parent.resolve() / "blocks_data.json") -> dict:
        loaded_block_data = {}
        try:
            if data_path.exists():
                with open(data_path, "r") as file:
                    loaded_block_data = json.load(file)  
            else:
                raise FileNotFoundError(f"Plik {data_path} nie istnieje.")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Błąd podczas ładowania danych bloków: {e}")
        return loaded_block_data

    def get_default_blocks_data(self) -> dict:
        return self.load_blocks_data(self.default_data_file_path)

    def get_smoothness_list(self) -> dict:
        return {block_type: block_data["smoothness"] for block_type, block_data in self.block_data.items()}

    def get_block_data(self) -> dict:
        return self.block_data

    def get_nb_blocks(self) -> int:
        return self.nb_blocks

    def get_filter_mode_list(self) -> list[list[str]]:
        return self.filter_mode_list

    def add_to_filter_list(self, block_type: str, list_type: int) -> None:
        if 0 <= list_type <= 2:
            self.filter_mode_list[list_type].append(block_type)
        else:
            raise ValueError("Nieprawidłowy typ listy filtrowania. Użyj 0 (whitelist), 1 (blacklist), 2 (neutral).")

    def setup(self) -> Path:
        
        self.block_data = self.load_blocks_data()
        self.update_nb_blocks()
        return self.get_running_path()

def update_progress_decorator(increment):
    def decorator(func):
        def wrapper(*args, **kwargs):
            self = args[0]
            result = func(*args, **kwargs)
            self.update_progress(increment)
            return result
        return wrapper
    return decorator


class ImageTools:
    def __init__(self):
        pass

    def load_image(self, path):
        return Image.open(path)

    def scale_image(self, image: Image.Image, target_size):
        original_width, original_height = image.size

        # Sprawdzenie, czy skalować w dół, czy w górę
        if original_width > target_size[0] or original_height > target_size[1]:
            # Skalowanie w dół - użyj thumbnail
            image.thumbnail(target_size, Image.Resampling.LANCZOS)
            new_width, new_height = image.size
        else:
            # Skalowanie w górę - użyj resize
            scale_factor = min(target_size[0] / original_width, target_size[1] / original_height)
            new_width, new_height = (int(original_width * scale_factor), int(original_height * scale_factor))
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        return image, new_width, new_height

    def create_thumbnail(self, original_image: Image.Image, target_size):
        thumbnail = original_image.copy()
        thumbnail, width, height = self.scale_image(thumbnail, target_size)
        thumbnail = ImageTk.PhotoImage(thumbnail)
        return thumbnail, width, height


class ImageProcessor:
    def __init__(self, block_manager, state_manager, progress_callback):
        self.run_path = Path(__file__).parent.resolve()
        self.block_manager = block_manager
        self.state_manager = state_manager

        self.image_path: Path = self.state_manager.image_path.get()
        self.scale: float = self.state_manager.scale.get() 
        self.remove_background: bool = self.state_manager.remove_background.get()
        self.gradient: bool = self.state_manager.gradient.get()
        self.resolution: int = self.state_manager.resolution.get()

        self.progress_callback = progress_callback
        self.listed_blocks = self.block_manager.get_listed_blocks()

        self.progress_level = 0 
        self.block_images: Dict[str, Image.Image] = {}
        self.blocks_color_data = {}
        self.alpha_channel = {}
        self.width = 0
        self.height = 0

    
    def update_progress(self, progress: int):
        self.progress_level += progress
        if self.progress_callback:
            self.progress_callback(self.progress_level)

    def main(self) -> Image.Image:
        logging.warning(f"Tworzenie nowego obrazu na podstawie: {self.image_path}")
        start_time = timeit.default_timer()
        
        self.load_blocks_color_data()
        self.load_and_prepare_image()
        closest_blocks = self.find_closest_blocks()
        result_image = self.create_result_image(closest_blocks)

        end_time = timeit.default_timer()
        logging.info(f"Stworzono obraz końcowy. Proces zajął: {end_time - start_time} sekund")
        return result_image
    
    @update_progress_decorator(5)
    def load_blocks_color_data(self):
        logging.info("Przygotowywanie danych kolorów bloków...")
        self.blocks_color_data = np.array([self.listed_blocks[block] for block in self.listed_blocks.keys()])

    @update_progress_decorator(5)
    def load_and_prepare_image(self):
        logging.info(f"Ładowanie i przygotowywanie obrazu ze skalą: {self.scale}") # dodać resampling kiedyś
        img = Image.open(self.image_path)
        img = img.resize((round(img.width * self.scale), round(img.height * self.scale)), Image.Resampling.BICUBIC)
        self.alpha_channel = self.prepare_alpha_channel(img)
        img = img.convert("RGB")
        self.width, self.height = img.size
        self.prepared_image = img
       
    @update_progress_decorator(15)
    def find_closest_blocks(self) -> List[str]:
        logging.info("Znajdowanie najbliższych bloków dla pikseli obrazu...")
        tree = KDTree(self.blocks_color_data)
        pixels = np.asarray(self.prepared_image).reshape(-1, 3)
        closest_indices = tree.query(pixels, k=3)[1]
        closest_blocks = [list(self.listed_blocks.keys())[i[0]] for i in closest_indices]
        return closest_blocks
    
    @update_progress_decorator(10)
    def create_result_image(self, closest_blocks):
        logging.info("Tworzenie obrazu wynikowego...")
        result = Image.new("RGBA", (self.width * self.resolution, self.height * self.resolution))
        block_matrix = self.reshape_to_matrix(closest_blocks, self.width, self.height)
        self.load_block_images(set(closest_blocks))
        result = self.place_blocks(block_matrix, result)
        return result

    @update_progress_decorator(4)
    def prepare_alpha_channel(self, img: Image.Image) -> Union[np.ndarray, Dict]:
        logging.info(f"Pobieranie kanału alpha z opcjami: remove_bg: {self.remove_background}, gradient: {self.gradient}")
        alpha = {}
        if img.mode == "RGB":
            logging.warning("Obraz nie posiada kanału alpha. Gradient i usuwanie tła zostaną zignorowane.")
            self.gradient = False
            self.remove_background = False
        if self.gradient or self.remove_background:
            img = img.convert("RGBA")
            alpha = np.array(img)[:, :, 3]
            if self.remove_background:
                if self.gradient:
                    alpha[alpha < 10] = 0
                else:
                    alpha = np.where(alpha <= 100, 0, 255)
            elif not self.gradient:
                alpha[:,:] = 255
            
        return alpha

    @update_progress_decorator(5)
    def reshape_to_matrix(self, closest_blocks: List[str], width: int, height: int) -> List[List[str]]:
        logging.info("Przekształcanie listy najbliższych bloków do macierzy...")
        return [closest_blocks[i*width:(i+1)*width] for i in range(height)]
    
    @update_progress_decorator(5)
    def load_block_images(self, block_names: set) -> None:

        assets_path = self.block_manager.get_block_images_path()
        default_assets_path = self.block_manager.get_default_block_images_path()

        for block in block_names:
            block_path = assets_path / block
            try:
                block_image = Image.open(block_path)
                self.block_images[block] = block_image
            except FileNotFoundError:
                logging.warning(f"Brak obrazu bloku: {block}, używanie domyślnego.")
                if default_assets_path.exists():
                    block_image = Image.open(default_assets_path / block)
                    self.block_images[block] = block_image
                else:
                    logging.error(f"Brak domyślnego obrazu bloku: {block}")

    def place_blocks(self, block_matrix: List[List[str]], result: Image.Image) -> Image.Image:
        is_alpha_needed = self.gradient or self.remove_background
        for y in range(self.height):
            for x, item in enumerate(block_matrix[y]):
                if is_alpha_needed:
                    alpha_value = self.alpha_channel[y][x]
                    if alpha_value == 0:
                        continue
                    obraz = self.block_images[item].copy()
                    obraz.putalpha(alpha_value)
                else:
                    obraz = self.block_images[item]

                result.paste(obraz, (x * self.resolution, y * self.resolution))

            if self.height > 10 and (y+1) % (self.height //5 ) == 0:
                self.update_progress(10)
        return result
    
class StateManager:
    def __init__(self):
        # zmienne filtrów
        self.filters = {
            "top": tk.BooleanVar(value=True),
            "side": tk.BooleanVar(value=True),
            "bottom": tk.BooleanVar(value=True),
            "solid": tk.BooleanVar(value=True),
            "smoothness_value": tk.DoubleVar(value=0)
        }
        # Zmienne ogólne
        self.general = {
            "scale": tk.DoubleVar(value=0.5)
        }
        # Zmienne tekstur
        self.resourcepack = {
            "resourcepack": tk.StringVar(value=""),
            "resolution": tk.IntVar(value=16)
        }
        # Zmienne przezroczystości
        self.transparent = {
            "remove_bg": tk.BooleanVar(value=False),
            "gradient": tk.BooleanVar(value=False)
        }
        # Inne zmienne
        self.var_image_path = tk.StringVar(value="")
        self.var_state = tk.StringVar(value="Wybierz obraz i naciśnij przycisk 'Przetwórz'")

    @property
    def scale(self):
        return self.general["scale"]

    @property
    def resolution(self):
        return self.resourcepack["resolution"]

    @property
    def smoothness_value(self):
        return self.filters["smoothness_value"]

    @property
    def remove_background(self):
        return self.transparent["remove_bg"]

    @property
    def gradient(self):
        return self.transparent["gradient"]

    @property
    def resourcepack_name(self):
        return self.resourcepack["resourcepack"]

    @property
    def image_path(self):
        return self.var_image_path

    @property
    def state(self):
        return self.var_state

    def get_filter_face_mode(self):
        mode = []
        if self.filters["top"].get():
            mode.append("top")
        if self.filters["side"].get():
            mode.append("side")
        if self.filters["bottom"].get():
            mode.append("bottom")
        if self.filters["solid"].get():
            mode.append("solid")
        return mode
    
    @property
    def filter_mode(self):
        return {
            "top": self.filters["top"],
            "side": self.filters["side"],
            "bottom": self.filters["bottom"],
            "solid": self.filters["solid"]
        }
        
    
WINDOW_WIDTH = 1730
WINDOW_HEIGHT = 935
CANVAS_SIZE = 510
PADDING = 5
BUTTONS_PADDING = 10

class Application:
    def __init__(self, root):
        self.root = root
        self.root.title("Drawcraft")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")

        self.state_manager = StateManager()
        self.block_manager = BlockManager(self.state_manager)
        self.image_tools = ImageTools()

        self.running_path = self.block_manager.setup()
        
        self.create_widgets()

    def create_widgets(self):
        # Czerwone pole - podgląd wybranego obrazu
        self.original_image_canvas = tk.Canvas(self.root, bg="red", relief="solid", height=CANVAS_SIZE, width=CANVAS_SIZE)
        self.original_image_canvas.grid(row=0, column=0, padx=PADDING, pady=PADDING, sticky="nw")
        self.original_image_canvas.grid_propagate(False)

        # Fioletowe pole - przyciski (Wybierz obraz, Przetwórz, Zapisz)
        self.purple_frame = tk.Frame(self.root, bg="purple", relief="solid", height=510, width=250)
        self.purple_frame.grid(row=0, column=1, padx=PADDING, pady=PADDING, sticky="new")
        self.purple_frame.grid_propagate(False)

        # Niebieskie pole - podgląd wygenerowanego obrazu
        self.processed_image_canvas = tk.Canvas(self.root, bg="blue", relief="solid", height=CANVAS_SIZE, width=CANVAS_SIZE)
        self.processed_image_canvas.grid(row=0, column=2, padx=PADDING, pady=PADDING, sticky="ne")
        self.processed_image_canvas.grid_propagate(False)

        # Zielone pole - ustawienia parametrów
        self.settings_frame = tk.Frame(self.root, height=400, width=1260)
        self.settings_frame.grid(row=1, column=0, columnspan=3, padx=PADDING, pady=PADDING, sticky="sew")
        self.settings_frame.grid_propagate(False)

        # Białe pole - progres bar
        self.progress_frame = tk.Frame(self.root, bg="white", relief="solid", height=800, width=400)
        self.progress_frame.grid(row=0, column=3, columnspan=1,rowspan=3, padx=PADDING, pady=PADDING, sticky="ne")
        self.progress_frame.grid_propagate(False)

        self.create_purple_frame()
        self.create_notebook()
        self.create_progress_frame()

    def create_purple_frame(self):
        # Ustawienia siatki dla button_frame
        self.purple_frame.grid_rowconfigure(0, weight=1)
        self.purple_frame.grid_rowconfigure(1, weight=1)
        self.purple_frame.grid_rowconfigure(2, weight=1)
        self.purple_frame.grid_columnconfigure(0, weight=1)

        # Przyciski rozciągnięte na całe fioletowe pole
        self.select_image_button = ttk.Button(self.purple_frame, text="Wybierz obraz", command=self.select_image, state="normal")
        self.select_image_button.grid(row=0, column=0, padx=BUTTONS_PADDING, pady=BUTTONS_PADDING, sticky="nsew")

        self.process_button = ttk.Button(self.purple_frame, text="Przetwórz obraz", command=self.process_image, state="disabled")
        self.process_button.grid(row=1, column=0, padx=BUTTONS_PADDING, pady=BUTTONS_PADDING, sticky="nsew")

        self.save_button = ttk.Button(self.purple_frame, text="Zapisz obraz", command=self.save_image, state="disabled")
        self.save_button.grid(row=2, column=0, padx=BUTTONS_PADDING, pady=BUTTONS_PADDING, sticky="nsew")
    
    def create_progress_frame(self):
        self.progress_bar = ttk.Progressbar(self.progress_frame, orient="horizontal", length=380, mode="determinate")
        self.progress_bar.grid(row=0, column=0, padx=PADDING, pady=PADDING, sticky="ew")

        label_progress_bar = tk.Label(self.progress_frame, textvariable=self.state_manager.state, font=("Arial", 10), bg="limegreen")
        label_progress_bar.grid(row=1, column=0, padx=PADDING, pady=PADDING, sticky="w")

    def create_tab(self, notebook, name, bg_color):
        tab = tk.Frame(notebook, bg=bg_color)
        notebook.add(tab, text=name)
        return tab
    
    def create_notebook(self):
        # Zakładki
        notebook = ttk.Notebook(self.settings_frame)
        notebook.pack(expand=True, fill='both')

        # Tworzenie zakładek
        tab_general = self.create_tab(notebook, "Główne", "white")
        tab_filters = self.create_tab(notebook, "Filtry", "white")
        tab_transparent = self.create_tab(notebook, "Przezroczystość", "white")
        tab_resourcepack = self.create_tab(notebook, "Resourcepack", "white")

        # Zakładka "Główne"    
        label_skala = tk.Label(tab_general, text="Skala:", font=("Arial", 20, "bold"), bg="limegreen")
        label_skala.grid(row=0, column=0, padx=BUTTONS_PADDING, pady=BUTTONS_PADDING, sticky="wn")
        
        entry_scale = ttk.Entry(tab_general,textvariable=self.state_manager.scale)
        entry_scale.grid(row=1, column=0, padx=BUTTONS_PADDING, pady=BUTTONS_PADDING, sticky="wn")

        # Zakładka "Filtry"
        label_tryb = tk.Label(tab_filters, text="Tryb:", font=("Arial", 20, "bold"), bg="limegreen")
        label_tryb.grid(row=0, column=0, padx=BUTTONS_PADDING, pady=BUTTONS_PADDING, sticky="w")

        check_top = tk.Checkbutton(tab_filters, text="top", variable=self.state_manager.filter_mode["top"], bg="limegreen", font=("Arial", 12))
        check_top.grid(row=1, column=0, padx=BUTTONS_PADDING, pady=BUTTONS_PADDING, sticky="wn")

        check_side = tk.Checkbutton(tab_filters, text="side", variable=self.state_manager.filter_mode["side"], bg="limegreen", font=("Arial", 12))
        check_side.grid(row=2, column=0, padx=BUTTONS_PADDING, pady=BUTTONS_PADDING, sticky="wn")

        check_bottom = tk.Checkbutton(tab_filters, text="bottom", variable=self.state_manager.filter_mode["bottom"], bg="limegreen", font=("Arial", 12))
        check_bottom.grid(row=3, column=0, padx=BUTTONS_PADDING, pady=BUTTONS_PADDING, sticky="wn")

        check_solid = tk.Checkbutton(tab_filters, text="solid", variable=self.state_manager.filter_mode["solid"], bg="limegreen", font=("Arial", 12))
        check_solid.grid(row=4, column=0, padx=BUTTONS_PADDING, pady=BUTTONS_PADDING, sticky="wn")

        label_smoothness = tk.Label(tab_filters, text="Gładkość:", font=("Arial", 20, "bold"), bg="limegreen")
        label_smoothness.grid(row=0, column=1, padx=BUTTONS_PADDING, pady=BUTTONS_PADDING, sticky="w")

        entry_smoothness_value = ttk.Entry(tab_filters,textvariable=self.state_manager.smoothness_value)
        entry_smoothness_value.grid(row=1, column=1, padx=BUTTONS_PADDING, pady=BUTTONS_PADDING, sticky="wn")

        # Zakładka "Przezroczystość"
        label_transparent = tk.Label(tab_transparent, text="Przezroczystość:", font=("Arial", 20, "bold"), bg="limegreen")
        label_transparent.grid(row=0, column=0, padx=BUTTONS_PADDING, pady=BUTTONS_PADDING, sticky="w")

        check_remove_bg = tk.Checkbutton(tab_transparent, text="Usuń tło", variable=self.state_manager.remove_background, bg="limegreen", font=("Arial", 12))
        check_remove_bg.grid(row=1, column=0, padx=BUTTONS_PADDING, pady=BUTTONS_PADDING, sticky="wn")

        check_gradient = tk.Checkbutton(tab_transparent, text="Gradient", variable=self.state_manager.gradient, bg="limegreen", font=("Arial", 12))
        check_gradient.grid(row=2, column=0, padx=BUTTONS_PADDING, pady=BUTTONS_PADDING, sticky="wn")

        # Zakładka "Resourcepack"
        label_resourcepack = tk.Label(tab_resourcepack, text="Resourcepack:", font=("Arial", 20, "bold"), bg="limegreen")
        label_resourcepack.grid(row=0, column=0, padx=BUTTONS_PADDING, pady=BUTTONS_PADDING, sticky="w")

        button_select_resourcepack = tk.Button(tab_resourcepack, text="Wybierz resourcepacka", font=("Arial", 12), bg="limegreen", command=self.select_resourcepack)
        button_select_resourcepack.grid(row=1, column=0, padx=BUTTONS_PADDING, pady=BUTTONS_PADDING, sticky="w")

        label_selected_rp = tk.Label(tab_resourcepack, textvariable=self.state_manager.resourcepack_name, font=("Arial", 8, "bold"), wraplength=172, bg="limegreen")
        label_selected_rp.grid(row=2, column=0, padx=BUTTONS_PADDING, pady=BUTTONS_PADDING, sticky="w")

        button_reset_resourcepack = tk.Button(tab_resourcepack, text="Resetuj resourcepacka", font=("Arial", 12), bg="limegreen", command=self.reset_resourcepack)
        button_reset_resourcepack.grid(row=3, column=0, padx=BUTTONS_PADDING, pady=BUTTONS_PADDING, sticky="w")

        label_resolution = tk.Label(tab_resourcepack, text="Rozdzielczość:", font=("Arial", 12), bg="limegreen")
        label_resolution.grid(row=0, column=1, padx=BUTTONS_PADDING, pady=BUTTONS_PADDING, sticky="w")

        entry_resolution = ttk.Entry(tab_resourcepack, textvariable=self.state_manager.resolution)
        entry_resolution.grid(row=1, column=1, padx=BUTTONS_PADDING, pady=BUTTONS_PADDING, sticky="wn")

    def select_image(self):
        file_path = filedialog.askopenfilename(initialdir=(self.running_path / "images"), filetypes=[("Image files", "*.jpg *.png")])
        if file_path:
            self.state_manager.image_path.set(file_path)
            img = self.image_tools.load_image(file_path)
            self.selected_img,width,height = self.image_tools.create_thumbnail(img, (512, 512))
            self.original_image_canvas.create_image((512-width)/2, (512-height)/2, anchor="nw", image=self.selected_img)
            self.process_button.config(state="normal")
    
    def reset_resourcepack(self):
        self.state_manager.resourcepack_name.set("")
        self.block_manager.reset_resourcepack()

    def select_resourcepack(self):
        file_path = filedialog.askopenfilename(initialdir=(self.running_path / "custom_resourcepacks"), filetypes=[("zip", "*.zip")])
        if file_path:
            self.state_manager.resourcepack_name.set(file_path)
            self.block_manager.set_resourcepack(Path(file_path))

    def set_processed_image_thumbnail(self):                
        width,height = full_result_image.size
        response = True
        if width * height > 12000*12000:
            response = messagebox.askokcancel (
                "Ostrzeżenie", 
                "Obraz został pomyślnie wygenerowany, ale jest zbyt duży, aby stworzyć jego podgląd. "
                "Czy chcesz kontynuować próbę stworzenia miniatury?\n\n"
                "Wybierz 'Ok', aby kontynuować tworzenie miniatury.\n"
                "Wybierz 'Anuluj', aby anulować tylko próbę tworzenia miniatury. Wygenrowany obraz nadal będzie można zapisać."
            )           
        if response:
            self.state_manager.state.set("Trwa tworzenie miniatury, może to zająć chwilę...")
            self.result_tk, width, height = self.image_tools.create_thumbnail(full_result_image, (512,512))
            self.processed_image_canvas.create_image((512-width)/2, (512-height)/2, anchor="nw", image=self.result_tk)
            self.state_manager.state.set("Obraz przetworzony pomyślnie.")
        else:
            self.state_manager.state.set("Obraz został wygenerowany. Miniatura nie została stworzona.")

    def process_image(self):
        self.progress_bar['value'] = 0  # Resetowanie paska postępu z poprzedniego generowania
        threading.Thread(target=self._process_image_task).start()

    def _process_image_task(self):
        global full_result_image

        self.state_manager.state.set("Rozpoczęto przetwarzanie obrazu...")
        self.update_buttons_state("disabled")
        self.block_manager.create_block_list()

        if not self.state_manager.image_path:
            messagebox.showerror("Błąd", "Proszę wybrać obraz.")
            self.update_buttons_state(state="error")
            return
        
        try:
            image_processor = ImageProcessor(self.block_manager, self.state_manager,progress_callback=self.progress_callback)
            full_result_image = image_processor.main()
            if full_result_image is None:
                raise Exception("Błąd w przetwarzaniu obrazu. Obraz nie został wygenerowany.")
            self.set_processed_image_thumbnail()
            self.progress_bar['value'] = 100
            self.update_buttons_state("normal")
            
        except Exception as e:
            messagebox.showerror("Błąd", str(e))
            self.update_buttons_state(state="error")

    def update_buttons_state(self, state):
        if state == "error":
            self.process_button.config(state="disabled")
            self.save_button.config(state="disabled")
            self.select_image_button.config(state="normal")
        else:
            self.process_button.config(state=state)
            self.save_button.config(state=state)
            self.select_image_button.config(state=state)
        
    def progress_callback(self, progress):
        self.progress_bar['value'] = progress
        self.progress_bar.update_idletasks()

    def save_image(self):
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if save_path:
            try:
                full_result_image.save(save_path)
                messagebox.showinfo("Sukces", "Obraz zapisany pomyślnie.")
            except Exception as e:
                messagebox.showerror("Błąd", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = Application(root)
    root.mainloop()
