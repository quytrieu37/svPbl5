# Generated by Django 4.0.4 on 2022-06-24 02:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('pbl5_api', '0003_uploadimagetest_disease_uploadimagetest_overview_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='uploadimagetest',
            name='imageSimiler',
            field=models.CharField(max_length=100, null=True),
        ),
    ]
