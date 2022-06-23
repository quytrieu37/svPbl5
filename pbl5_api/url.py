
from django.urls import path
from .views import ImageViewSet,PridictLast
urlpatterns = [
    path('upload/', ImageViewSet.as_view()),
    path('pridict-last/', PridictLast.as_view()),
]
