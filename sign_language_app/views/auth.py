from django.shortcuts import render, redirect
from sign_language_app.forms import NewUserForm
from django.contrib.auth import login
from django.contrib import messages


def register_account(request):
	if request.method == "POST":
		form = NewUserForm(request.POST)
		print(form.errors)
		if form.is_valid():
			user = form.save()
			login(request, user)
			messages.success(request, "Registration successful.")
			return redirect("index")
		else:
			return render(request=request, template_name="sign_language_app/auth/register.html", context={"form": form})
	else:
		# messages.error(request, "Registration failed.")
		form = NewUserForm()
		return render(request=request, template_name="sign_language_app/auth/register.html", context={"form": form})