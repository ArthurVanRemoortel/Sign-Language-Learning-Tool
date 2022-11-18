from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework import status
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.response import Response
from rest_framework.parsers import JSONParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import SessionAuthentication, BasicAuthentication
from pprint import pprint


@api_view(['POST', 'OPTIONS'])
@authentication_classes([SessionAuthentication, BasicAuthentication])
@permission_classes([IsAuthenticated])
def test_auth(request):
    data = request.data
    left = data['left_landmarks']
    right = data['right_landmarks']

    print(f'Left: {len(left["1"])}')
    print(f'Right: {len(right["1"])}')
    return JsonResponse({'status': 'OK'}, status=status.HTTP_201_CREATED)
