from django.shortcuts import render


def error_view(request, exception, status=404):
    if isinstance(exception, Exception):
        error_message = exception
    else:
        error_message = exception

    context = {"request": request, "error_message": error_message}
    return render(request, "sign_language_app/errors/404.html", context, status=status)
