# Django imports
from django.contrib import admin
from django.urls import path, include


# URLs
urlpatterns = [
    path('ia/', include('core.IA.urls'))
]
