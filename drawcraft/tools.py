import os
import shutil
from typing import List, Tuple
from pathlib import Path

class Tools():

    def delete_folder(self, path: str | Path) -> None:
        path = Path(path)
        for file_path in path.iterdir():
            try:
                if file_path.is_file() or file_path.is_symlink():
                    file_path.unlink()
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Nie udało się usunąć {file_path}. Powód: {e}')

    def get_all_img_paths(self, path: str | Path) -> List[Path]:
        path = Path(path)  # Konwersja do Path, jeśli podano str
        return [
            file_path
            for file_path in path.rglob("*")  # Rekurencyjnie przechodzi przez wszystkie pliki
            if file_path.suffix.lower() in [".jpg", ".jpeg", ".png"]  # Filtrujemy po rozszerzeniach plików
        ]