from django.db import models
from django.contrib.auth.models import AbstractUser

# Create your models here.
def nameFile(instance, filename):
    return '/'.join(['images', str(instance.name), filename])

class UploadImageTest(models.Model):
    name = models.CharField(max_length=100)
    image = models.ImageField(upload_to=nameFile, blank=True, null=True)
    createAt = models.DateTimeField(null=True)
    disease = models.CharField(max_length=100, null=True)
    plantName = models.CharField(max_length=100, null=True)
    overview = models.CharField(max_length=1000,null=True)
    solutions = models.CharField(max_length=2000,null=True)
    imageSimilar = models.CharField(max_length=100,null=True)
    predictAt = models.DateTimeField(null=True)


class User(AbstractUser):
    # Delete not use field
    username = None
    last_login = None
    is_staff = None
    is_superuser = None

    password = models.CharField(max_length=100)
    email = models.EmailField(max_length=100, unique=True)
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []

    def __str__(self):
        return self.email