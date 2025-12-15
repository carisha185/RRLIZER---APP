from django.shortcuts import render

def home(request):
    return render(request, 'home.html')

def about(request):
    return render(request, 'about.html')

def help(request):
    return render(request, 'help.html')

def preprocess(request):
    return render(request, 'preprocess.html')

def sentiment(request):
    return render(request, 'sentiment.html')

from django.http import HttpResponse

def home(request):
    return HttpResponse('sentiment_app/base.html')
