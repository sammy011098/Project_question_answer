from django.urls import path,include

from question_answerr import views

urlpatterns = [
    path('', views.user_login, name='user_login'),
    path('send_data', views.send_data, name='send_data')
]