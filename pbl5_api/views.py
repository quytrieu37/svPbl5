import json
from sys import api_version
from django.http import HttpResponse
from django.shortcuts import render
from PIL import Image
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework import permissions
from .models import UploadImageTest
from .serializer import ImageSerializer ,ResultPridictSerialize
from .predict import single_prediction, get_solution
from .result import Result
# Create your views here.
class ImageViewSet(APIView):
    queryset = UploadImageTest.objects.all()
    serializer_class = ImageSerializer
    def get(seft, request,*args, **kwargs):
        images = UploadImageTest.objects.filter()
        serializer = ImageSerializer(images, many = True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    def post(self, request, *args, **kwargs):
        file = request.data['image']
        image = UploadImageTest.objects.create(image=file)
        return HttpResponse(json.dumps({'message': "Uploaded"}), status=200)

class PridictLast(APIView):
    def get(seft, request,*args, **kwargs):
        images = UploadImageTest.objects.filter().order_by('-id')
        serializer = ImageSerializer(images, many = True)
        data = serializer.data
        rs = single_prediction(data[0]['image'])
        name = rs.split(' ',1)
        rss = get_solution(name[0], name[1])
        result = Result(name[0],name[1], rss["oveview"], rss["solution"], data[0]['image'], data[0]['id'])
        serializer = ResultPridictSerialize(result)
        return Response(serializer.data, status=status.HTTP_200_OK)