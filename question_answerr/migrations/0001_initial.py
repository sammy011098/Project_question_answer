# Generated by Django 3.2.8 on 2022-02-18 09:32

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Question_answerr',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('question', models.CharField(max_length=1000)),
                ('answer', models.CharField(max_length=5000)),
            ],
            options={
                'db_table': 'question_answerr',
            },
        ),
    ]
