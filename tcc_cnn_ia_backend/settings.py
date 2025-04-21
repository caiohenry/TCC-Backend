# Standard library imports
import os
from pathlib import Path

# Third-party imports
import datetime
from dotenv import load_dotenv

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Import environment variables
load_dotenv()


# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.getenv("DJANGO_SECRET_KEY")


# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.getenv("DJANGO_DEBUG")


# Allowed hosts
ALLOWED_HOSTS = [
    # TODO: ! REMOVE THIS "*" IN PRODUCTION
    "*",
    "127.0.0.1",
]


# CORS Settings (allow all)
CORS_ORIGIN_ALLOW_ALL = True


# CORS Settings (allow list)
CORS_ORIGIN_WHITELIST = []


# CORS Origin Settings (allow list)
SECURE_CROSS_ORIGIN_OPENER_POLICY = "same-origin"


# Application definition
INSTALLED_APPS = [

    # DJANGO APPS
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",

    # PACKAGES
    "rest_framework"


]


# Middlewares 
MIDDLEWARE = [

    # Security Middleware
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",

    # CORS Middleware
    "corsheaders.middleware.CorsMiddleware"

]


# ROOT URL
ROOT_URLCONF = 'tcc_cnn_ia_backend.urls'


# Templates
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]


# WSGI Application
WSGI_APPLICATION = "tcc_cnn_ia_backend.wsgi.application"


# ASGI Application
ASGI_APPLICATION = "tcc_cnn_ia_backend.asgi.application"


# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}


# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True


# Static files (CSS, JavaScript, Images)
STATIC_URL = 'static/'


# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
