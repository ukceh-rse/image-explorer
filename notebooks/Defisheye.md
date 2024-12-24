---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.6
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import numpy as np
from skimage import io
```

```python
def sample_image():
    # Create 400x400 RGB image
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    y, x = np.ogrid[-200:200, -200:200]
    r = np.sqrt(x*x + y*y)
    
    # Create concentric circles with different RGB colors
    colors = [
        [255, 0, 0],    # Red
        [0, 255, 0],    # Green
        [0, 0, 255],    # Blue
        [255, 255, 0],  # Yellow
        [0, 255, 255],  # Cyan
    ]
    
    for i, color in enumerate(colors):
        radius = 40 * (i + 1)
        mask = (r >= radius - 20) & (r < radius)
        image[mask] = color
    
    return image
```

```python
io.imshow(sample_image())
```

```python
import sys
sys.path.append('../src')
from phenocam.image.defisheye import do_defisheye
from phenocam.image.slice import slice_image_in_half
```

```python
do_defisheye(sample_image())
```

```python
i = do_defisheye(slice_image_in_half(io.imread('../tests/fixtures/WADDN_20140101_0902_ID405.jpg'))[0])
```

```python
i = do_defisheye(slice_image_in_half(io.imread('../tests/fixtures/WADDN_20240101_0910_ID20240101091001.jpg'))[0])
```

```python
i
```

```python
from skimage import transform
transform.resize(i,(600,600), preserve_range=True, anti_aliasing=True)
```

```python
io.imshow(do_defisheye(slice_image_in_half(io.imread('../tests/fixtures/WADDN_20140101_0902_ID405.jpg'))[0]))
```

```python
import thingsvision
```

```python
from thingsvision import get_extractor
from thingsvision.utils.storing import save_features
from thingsvision.utils.data import ImageDataset, DataLoader

model_name = 'simclr-rn50'
source = 'ssl'
device = 'cpu'

extractor = get_extractor(
  model_name=model_name,
  source=source,
  device=device,
  pretrained=True
)

```

```python
root='../tests/fixtures' # (e.g., './images/)
batch_size = 1

dataset = ImageDataset(
  root=root,
  out_path='../features',
  backend=extractor.get_backend(),
  transforms=extractor.get_transformations()
)

batches = DataLoader(
  dataset=dataset,
  batch_size=batch_size, 
  backend=extractor.get_backend()
)

```

```python
extractor.model

```

```python
module_name= 'layer4.2.conv3'
features = extractor.extract_features(
  batches=batches,
  module_name=module_name,
  flatten_acts=True  # flatten 2D feature maps from convolutional layer
)

save_features(features, out_path='features', file_format='npy')
```
