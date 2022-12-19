import os
import time
from pathlib import Path

from django import forms
from django.contrib.auth import authenticate, password_validation
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm, PasswordResetForm, SetPasswordForm
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.forms import ModelForm

from sign_language_app.models import Course


class NewUserForm(UserCreationForm):
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={
            "class": "input",
            "id": "emailInput",
            "placeholder": "Email",
            "autofocus": True
        }),
        required=True
    )

    username = forms.CharField(
        widget=forms.TextInput(attrs={
            "class": "input",
            "id": "usernameInput",
            "placeholder": "Username"
        }),
        required=True,
    )

    password1 = forms.CharField(
        label="Password",
        strip=False,
        widget=forms.PasswordInput(
            attrs={"class": "input",
                   "id": "passwordInput",
                   "placeholder": "Password",
                   "autocomplete": "new-password"
                   }
        ),
        required=True,
    )
    password2 = forms.CharField(
        label="Password Confirmation",
        strip=False,
        widget=forms.PasswordInput(
            attrs={"class": "input",
                   "id": "passwordInput",
                   "placeholder": "Password Confirmation",
                   "autocomplete": "new-password"
                   }
        ),
        required=True,
    )

    class Meta:
        model = User
        fields = ("username", "email", "password1", "password2")

    def save(self, commit=True):
        user = super(NewUserForm, self).save(commit=False)
        user.email = self.cleaned_data['email']
        if commit:
            user.save()
        return user


class LoginForm(forms.Form):
    # email = forms.EmailField(
    #     widget=forms.EmailInput(attrs={"class": "input", "id": "emailInput"}),
    #     required=True
    # )

    username = forms.CharField(
        widget=forms.TextInput(attrs={
            "class": "input",
            "id": "usernameInput",
            "placeholder": "Username",
            "autofocus": True
        }),
        required=True,
    )

    password = forms.CharField(
        label="Password",
        strip=False,
        widget=forms.PasswordInput(
            attrs={"class": "input",
                   "id": "passwordInput",
                   "placeholder": "Password",
                   # "autocomplete": "current-password"
                   }
        ),
        required=True,
    )

    class Meta:
        model = User
        fields = ("username", "password")

    def __init__(self, request=None, *args, **kwargs):
        self.request = request
        self.user_cache = None
        super().__init__(*args, **kwargs)

    def clean(self):
        username = self.cleaned_data.get("username")
        password = self.cleaned_data.get("password")
        if username is not None and password:
            self.user_cache = authenticate(
                self.request, username=username, password=password
            )
            if self.user_cache is None:
                raise ValidationError(
                    "Please enter a correct username and password. Note that both fields may be case-sensitive."
                )
        return self.cleaned_data

    def get_user(self):
        return self.user_cache


class NewPasswordAuthenticationForm(SetPasswordForm):
    def __init__(self, *args, **kwargs):
        super(NewPasswordAuthenticationForm, self).__init__(*args, **kwargs)

    new_password1 = forms.CharField(
        label="New password",
        widget=forms.PasswordInput(attrs={
            "autocomplete": "new-password",
            "class": "input",
            "placeholder": "New Password",
            "autofocus": True

        }),
        strip=False,
        help_text=password_validation.password_validators_help_text_html()
    )
    new_password2 = forms.CharField(
        label="New password confirmation",
        strip=False,
        widget=forms.PasswordInput(attrs={
            "autocomplete": "new-password",
            "class": "input",
            "placeholder": "Confirm Password"
        }),
    )


class ResetPasswordAuthenticationForm(PasswordResetForm):
    def __init__(self, *args, **kwargs):
        super(ResetPasswordAuthenticationForm, self).__init__(*args, **kwargs)

    email = forms.CharField(
        label="Email",
        widget=forms.EmailInput(attrs={
            "autocomplete": "email",
            "class": "input",
            "placeholder": "Email",
            "autofocus": True
        }),
        max_length=254
    )


class CoursesForm(forms.Form):
    search_input = forms.CharField(
        widget=forms.TextInput(attrs={"class": "input", "id": "searchInput"}),
        required=False,
    )



class UploadGestureForm(forms.Form):
    word = forms.CharField(
        widget=forms.TextInput(attrs={"class": "input", "id": "wordInput"}),
        required=True,
    )
    left_hand = forms.BooleanField(
        widget=forms.CheckboxInput(attrs={"id": "leftHandInput"}),
        required=False
    )
    right_hand = forms.BooleanField(
        widget=forms.CheckboxInput(attrs={"id": "rightHandInput"}),
        required=False
    )
    videos = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': True, 'id': 'videosInput', 'class': 'file-input'}))

    def handle_uploaded_files(self, request, user):
        # TODO: Check for max file size
        word = self.cleaned_data['word']
        left_hand = self.cleaned_data['left_hand']
        right_hand = self.cleaned_data['right_hand']
        root_path = Path('sl_ai/ai_data/vgt-uploaded') / str(user.id) / f"{word}_{1 if left_hand else 0}{1 if right_hand else 0}"
        root_path.mkdir(parents=True, exist_ok=True)
        for file in request.FILES.getlist('videos'):
            with open(root_path / file.name, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)


class NewCourseForm(forms.Form):
    title = forms.CharField(
        widget=forms.TextInput(attrs={
            "class": "input",
            "id": "titleInput",
        }),
        required=True,
    )
    description = forms.CharField(
        widget=forms.Textarea(attrs={
            "class": "textarea",
            "id": "descriptionInput",
            'rows': 6, 'cols': 40,
            "placeholder": "Please explain to the uses what this course is about."
        }),
        required=True,
    )
    visibility = forms.ChoiceField(
        required=False,
        choices=Course.Visibility.choices,
    )

    difficulty = forms.ChoiceField(
        required=False,
        choices=Course.Difficulty.choices,
    )