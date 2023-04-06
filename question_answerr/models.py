from django.db import models

# Create your models here.
class Question_answerr(models.Model):
    id = models.AutoField(primary_key=True)
    question = models.CharField(max_length=1000)
    answer = models.CharField(max_length=5000)
    class Meta:
        db_table = "question_answerr"
