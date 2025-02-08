# Below code generated by Perplexity and modified: https://www.perplexity.ai/search/can-you-give-me-an-end-to-end-nvd2eA.9RSCq_JuTiyrp.A#2
import torch
import torchvision.models as models

# Load pretrained ResNet50 model
model = models.resnet50(pretrained=True)

# Save the model
torch.save(model.state_dict(), './models/toy_resnet50.pth')

print('Pretrained ResNet50 model saved successfully')
