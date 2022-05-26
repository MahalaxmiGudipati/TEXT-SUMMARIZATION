from django.shortcuts import render,redirect
from django.http import HttpResponse
from django.forms import inlineformset_factory
from django.contrib.auth.forms import UserCreationForm
from .forms import CreateUserForm

from django.contrib import messages
from django.contrib import messages
from django.contrib.auth import authenticate,login,logout
from django.contrib.auth.decorators import login_required

def home(request):
	return render(request, 'index.html')
    
def loginPage(request):
    if request.method=='POST':
        username=request.POST.get('username')
        password=request.POST.get('password')
            
        user=authenticate(request, username=username, password=password)
        if user is not None:
            login(request,user)
            return redirect('http://localhost:8501')
        else:
            messages.info(request,'Username OR  password is incorrect')
            
        
    context={}
    return render(request, 'login.html',context)

def register(request):
    form = CreateUserForm()

    if request.method=='POST':
        form=CreateUserForm(request.POST)
        if form.is_valid():
            form.save()
            user=form.cleaned_data.get('username')
            messages.success(request, 'account created for ' +user)
            return redirect('login')

    return render(request, 'register.html')

def logoutUser(request):
    logout(request)
    return redirect('login')


def register(request):
    form = CreateUserForm()

    if request.method=='POST':
        form=CreateUserForm(request.POST)
        if form.is_valid():
            form.save()
            user=form.cleaned_data.get('username')
            messages.success(request,'Account was created for '+ user)
            return redirect('login')

    context={'form': form}
    return render(request, 'register.html',context)
def summary(request):
	return render(request, 'summary.html')
