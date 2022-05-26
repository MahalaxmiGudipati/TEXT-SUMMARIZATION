

from django.urls import path
from . import views


urlpatterns = [
    path('',views.home,name='index'),
    path('login/',views.loginPage,name='login'),
    path('register/',views.register,name='register'),
    path('logout/', views.logoutUser,name="logout"),
    path('summary/', views.summary,name="summary"),
    
    
]
