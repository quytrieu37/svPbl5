from rest_framework import serializers
from .models import UploadImageTest ,User
from .result import Result

class ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = UploadImageTest
        fields = ('id','name', 'image','createAt', 'plantName','disease','overview','solutions','imageSimilar', 'predictAt')

class ResultPridictSerialize(serializers.Serializer):
    treeName = serializers.CharField()
    disease = serializers.CharField()
    oveview = serializers.CharField()
    solution = serializers.CharField()
    imageName = serializers.CharField(max_length=200)
    imageId = serializers.CharField()
    imageSimilar = serializers.CharField()
    predictDate = serializers.CharField()
    # solution = serializers.CharField()
class UserSerializer(serializers.ModelSerializer):

    class Meta:
        model = User
        fields = ('email', 'password')
        extra_kwargs = {'password': {'write_only': True}}