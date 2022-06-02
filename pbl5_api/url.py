
from django.urls import path
from .views import ImageViewSet,GetImage

urlpatterns = [
    path('upload/', ImageViewSet.as_view()),
    path('images/', GetImage.as_view()),
]
