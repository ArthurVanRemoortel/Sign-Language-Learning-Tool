from django.shortcuts import render, redirect
from sign_language_app.forms import NewUserForm, LoginForm
from django.contrib.auth import login, authenticate
from django.contrib import messages


def register_account(request):
    if request.method == "POST":
        form = NewUserForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            # messages.success(request, "Registration successful.")
            return redirect("index")
        else:
            return render(
                request=request,
                template_name="sign_language_app/auth/register.html",
                context={"form": form},
            )
    else:
        # messages.error(request, "Registration failed.")
        form = NewUserForm()
        return render(
            request=request,
            template_name="sign_language_app/auth/register.html",
            context={"form": form},
        )


def login_account(request):
    if request.method == "POST":
        form = LoginForm(request, request.POST)
        if form.is_valid():
            user = form.get_user()
            if user is not None:
                login(request, user)
                return redirect("index")
        return render(
            request=request,
            template_name="sign_language_app/auth/login.html",
            context={"form": form},
        )

    else:
        form = LoginForm()
        return render(
            request=request,
            template_name="sign_language_app/auth/login.html",
            context={"form": form},
        )
