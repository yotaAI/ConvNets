
# Neural Style Transfer
@yota
Neural style transfer is an optimization technique used to take two images—a content image and a style reference image (such as an artwork by a famous painter)—and blend them together so the output image looks like the content image, but “painted” in the style of the style reference image.

Note : 

🏷️ I tried both Adam and L-BFGS optimzier.

🏷️Adam Optimizer is having good output after 200 epoch.

🏷️L-BFGS after 2nd iterationn style is merging on the image but The output is coming bad.

## Tests

#### Images


| Original  |    Style     |   Adam [200 Steps]   |  L-BFGS [2 Steps]   |
| :-------- | :------- | :------------------------- |:-------- |
| <img src="https://github.com/yotaAI/ConvNets/assets/38225850/b744f273-2943-48d3-9f39-29569ff4d221" alt="J" width="250"/> | <img src="https://github.com/yotaAI/ConvNets/assets/38225850/96683204-0afe-4040-97a6-bf88f71ac76b" alt="J" width="250"/> | <img src="https://github.com/yotaAI/ConvNets/assets/38225850/1d5f98f2-ca7d-4e00-9e82-b49fa6615274" alt="J" width="250"/> | <img src="https://github.com/yotaAI/ConvNets/assets/38225850/6d01aa01-4ec5-4d02-b994-3188cbbe4bd0" alt="J" width="250"/> |



