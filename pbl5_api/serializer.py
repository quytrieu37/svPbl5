from rest_framework import serializers
from .models import UploadImageTest
from .result import Result

class ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = UploadImageTest
        fields = ('id','name', 'image')

class ResultPridictSerialize(serializers.Serializer):
    treeName = serializers.CharField()
    disease = serializers.CharField()
    imageName = serializers.CharField(max_length=200)
    imageId = serializers.CharField()
    # solution = serializers.CharField()