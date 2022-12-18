from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db.models import Q
from django.shortcuts import render, redirect

from sign_language_app.background_tasks import retrain_model
from sign_language_app.forms import UploadGestureForm
from sign_language_app.models import Gesture
from sign_language_app.views.views import get_user


@login_required
def profile_overview(request):
    user = get_user(request)
    context = {
        'current_section': 'overview'
    }
    return render(request, "sign_language_app/profile/profile_overview.html", context)


@login_required
def profile_settings(request):
    user = get_user(request)
    context = {
        'current_section': 'settings'
    }
    return render(request, "sign_language_app/profile/profile_overview.html", context)




@login_required
def manage_students(request):
    user = get_user(request)
    context = {
        'current_section': 'manage_students'
    }
    return render(request, "sign_language_app/profile/manage_students.html", context)


@login_required
def manage_courses(request):
    user = get_user(request)
    context = {
        'current_section': 'manage_courses'
    }
    return render(request, "sign_language_app/profile/manage_courses.html", context)


@login_required
def manage_gestures(request):
    user = get_user(request)
    upload_gesture_form = UploadGestureForm(request.GET)
    # retrain_model()
    context = {
        'current_section': 'manage_gestures',
        'upload_gesture_form': upload_gesture_form,
        'gestures': Gesture.objects.filter(Q(creator=user)).all()
    }
    return render(request, "sign_language_app/profile/manage_gestures.html", context)


@login_required
def create_gesture(request):
    user = get_user(request)
    if request.method == 'POST':
        # for video_file in request.FILES.getlist('mkv'):
        form = UploadGestureForm(request.POST, request.FILES)
        if form.is_valid():
            gesture_word = form.cleaned_data.get('word')
            left_hand = form.cleaned_data.get('left_hand')
            right_hand = form.cleaned_data.get('right_hand')
            existing_gesture = Gesture.objects.filter(Q(word__iexact=gesture_word) & Q(creator=user)).first()
            if existing_gesture:
                messages.error(request, 'You already created a gesture for that word.')
                return redirect('manage_gestures')
            if not left_hand and not right_hand:
                messages.error(request, 'Please specify which hands are used to perform the gesture')
                return redirect('manage_gestures')

            form.handle_uploaded_files(request, user=user)
            new_gesture = Gesture(word=gesture_word, left_hand=left_hand, right_hand=right_hand, creator=user, status=Gesture.Status.PENDING)
            new_gesture.save()

            retrain_model()
        else:
            messages.error(request, 'The form was not valid. Please try again and make sure to fill in all the necessary information')
    return redirect('manage_gestures')


@login_required
def feedback(request):
    user = get_user(request)
    context = {
        'current_section': 'feedback'
    }
    return render(request, "sign_language_app/profile/feedback.html", context)
