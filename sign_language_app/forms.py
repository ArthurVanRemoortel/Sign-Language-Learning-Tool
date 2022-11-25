from django import forms
from django.contrib.auth import authenticate, password_validation
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm, PasswordResetForm, SetPasswordForm
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.forms import ModelForm


class NewUserForm(UserCreationForm):
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={
            "class": "input",
            "id": "emailInput",
            "placeholder": "Email",
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
                   # "autocomplete": "current-password"
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
                   # "autocomplete": "current-password"
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
            "placeholder": "Username"
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
            "placeholder": "New Password"
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
            "placeholder": "Email"
        }),
        max_length=254
    )


class CoursesForm(forms.Form):
    search_input = forms.CharField(
        widget=forms.TextInput(attrs={"class": "input", "id": "searchInput"}),
        required=False,
    )

