# Django imports
from django.urls import path
                    
# First-party imports
from .views import *

                       
# URL's for the IA
urlpatterns = [

    # Define a route for listing all objects of a certain class
    path('image/', IATestImage.as_view(), name='image-test'),

]