import os
from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader
from .oxford_pets import OxfordPets
from .dalle_pets import Dalle_Pets


class Dalle_Sun(DatasetBase):
    
    dataset_dir = 'dalle_sun397'

    def __init__(self, cfg, clean_data=False):
        root = cfg['root_path']
        num_shots = cfg['dalle_shots']
        open_world = cfg['open_world']
        num_classes = cfg['num_classes']

        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'SUN397')
        self.split_path = os.path.join(self.dataset_dir, 'dalle_sun.json')

        train = Dalle_Pets.read_split(self.split_path, self.image_dir)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, open_world=open_world, num_classes=num_classes)
