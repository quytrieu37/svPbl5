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
from django.shortcuts import get_object_or_404
import datetime
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
        now = datetime.datetime.now()
        image = UploadImageTest.objects.create(image=file,createAt = now)
        return HttpResponse(json.dumps({'message': "Uploaded"}), status=200)

class PridictLast(APIView):
    def get(seft, request,*args, **kwargs):
        images = UploadImageTest.objects.filter().order_by('-id')[0]
        serializer = ImageSerializer(images)
        data = serializer.data
        rs = single_prediction(data['image'])
        name = rs[0].split(' ',1)
        rss = get_solution(name[0], name[1])
        images.plantName = name[0]
        images.disease = name[1]
        images.overview = rss["oveview"]
        images.solutions = rss["solution"]
        images.imageSimilar = rs[1]
        images.predictAt = datetime.datetime.now()
        images.save()
        result = Result(name[0],name[1], rss["oveview"], rss["solution"], data['image'], data['id'], rs[1],datetime.datetime.now())
        serializer = ResultPridictSerialize(result)
        return Response(serializer.data, status=status.HTTP_200_OK)
class PridictImg(APIView):
    model = UploadImageTest
    serializer_class = ImageSerializer
    def get(seft, request,*args, **kwargs):
        id=kwargs.get('pk')
        image = UploadImageTest.objects.get(id=id)
        serializer = ImageSerializer(image)
        data = serializer.data
        rs = single_prediction(data['image'])
        name = rs[0].split(' ',1)
        rss = get_solution(name[0], name[1])
        image.plantName = name[0]
        image.disease = name[1]
        image.overview = rss["oveview"]
        image.solutions = rss["solution"]
        image.imageSimilar = rs[1]
        image.predictAt = datetime.datetime.now()
        image.save()
        result = Result(name[0],name[1], rss["oveview"], rss["solution"], data['image'], data['id'], rs[1],datetime.datetime.now())
        serializer = ResultPridictSerialize(result)
        return Response(serializer.data, status=status.HTTP_200_OK)
class GetDataOne(APIView):
    model = UploadImageTest
    serializer_class = ImageSerializer
    def get(seft, request,*args, **kwargs):
        image = get_object_or_404(UploadImageTest, id=kwargs.get('pk'))
        serializer = ImageSerializer(image)
        data = serializer.data
        result = Result(data['plantName'],data['disease'], data['overview'], data['solutions'], data['image'], data['id'], data['imageSimilar'],data['predictAt'])
        serializer = ResultPridictSerialize(result)
        return Response(serializer.data, status=status.HTTP_200_OK)