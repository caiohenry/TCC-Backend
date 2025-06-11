# Django imports
from rest_framework import generics, response, status
from django.core.files.storage import default_storage
from django.utils.translation import gettext_lazy as _
from django.core.files.base import ContentFile
from django.conf import settings

# First-party imports
from core.IA.model_cnn import CNN
from core.IA.variables import image_transform

# Standard library imports
import torch
from torchvision import transforms as T
from PIL import Image
import os


# POST Method
class IATestImage(generics.GenericAPIView):

    # Deactivate authentication
    authentication_classes = ()

    # Overriding the method
    def post(self, request):

        from time import sleep
        sleep(1)

        # Classes names
        class_names = ['Possui Câncer', 'Não Possui Câncer'] 

        # CNN Model Instance	
        cnn = CNN()

        # Loading model
        cnn.load_state_dict(torch.load('/home/caio/Área de Trabalho/TCC/tcc_cnn_ia_backend/core/IA/models/acuracia_97_model_3.pth'))

        # Putting model into evaluation mode
        cnn.eval()

        # Get image file from request body
        image_file = request.FILES['image']
        
        # Temporarily save the image
        temp_path = default_storage.save('temp/' + image_file.name, ContentFile(image_file.read()))
        full_temp_path = os.path.join(default_storage.location, temp_path)
        
        # Load and transform the image
        image = Image.open(full_temp_path).convert('RGB')
        image_tensor = image_transform(image).unsqueeze(0)
        
        # Make the prediction
        with torch.no_grad():
            outputs = cnn(image_tensor)
            _, predicted = torch.max(outputs.data, 1)
            predicted_class = class_names[predicted.item()]
        
        # Clear temporary image
        default_storage.delete(temp_path)
        
        # Return data
        return response.Response(data = {"predicted_class": predicted_class}, status = status.HTTP_200_OK)



