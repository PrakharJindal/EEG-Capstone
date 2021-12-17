# todos/urls.py
from django.urls import path, include
from . import views
from django.conf.urls.static import static
from django.conf import settings


urlpatterns = [
    path('', views.home, name='home'),
    path('Home.html', views.home, name='home2'),
    path('Contact.html', views.contact, name='contact'),
    path('Predict.html', views.predict, name='predict'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)   # FOR IMAGES
