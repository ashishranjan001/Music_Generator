from django.shortcuts import render
from Home.LSTM_Attention_LSTM.predict_CPU import generate

# Create your views here.


def index(request):
    return render(request, 'index.html')

def generateSong(request):
	generate()
	return render(request, 'index.html')