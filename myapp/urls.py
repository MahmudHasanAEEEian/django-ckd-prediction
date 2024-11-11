from django.contrib import admin
from django.urls import path, include
from myapp import views

app_name = 'myapp' 

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.predict_view, name='predict_view'),  # Direct root URL to predict_view
    # other paths...
]
