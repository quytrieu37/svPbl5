
from django.urls import path
from .views import ImageViewSet,PridictLast , PridictImg, GetDataOne, UserRegisterView
from rest_framework_simplejwt import views as jwt_views

urlpatterns = [
    path('upload/', ImageViewSet.as_view()),
    path('pridict-last/', PridictLast.as_view()),
    path('predict-one/<int:pk>', PridictImg.as_view()),
    path('image-detail/<int:pk>', GetDataOne.as_view()),
    path('register', UserRegisterView.as_view(), name='register'),
    path('login', jwt_views.TokenObtainPairView.as_view(), name='login'),
]
