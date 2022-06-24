
from django.urls import path
from .views import ImageViewSet,PridictLast , PridictImg, GetDataOne
urlpatterns = [
    path('upload/', ImageViewSet.as_view()),
    path('pridict-last/', PridictLast.as_view()),
    path('predict-one/<int:pk>', PridictImg.as_view()),
    path('image-detail/<int:pk>', GetDataOne.as_view()),
]
