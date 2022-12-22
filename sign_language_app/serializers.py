from django.contrib.auth.models import User, Group
from rest_framework import serializers

from sign_language_app.models import Gesture


class GestureSerializer(serializers.HyperlinkedModelSerializer):
    creator_name = serializers.SerializerMethodField('get_creator_name')
    creator_id = serializers.SerializerMethodField('get_creator_id')

    class Meta:
        model = Gesture
        depth = 1
        fields = ["id", "word", "left_hand", "right_hand", "status", "creator_name", "creator_id"]

    def get_creator_name(self, obj: Gesture):
        if obj.creator:
            return obj.creator.username
        return None

    def get_creator_id(self, obj: Gesture):
        if obj.creator:
            return obj.creator.id
        return None
