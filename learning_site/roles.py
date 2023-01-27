from rolepermissions.roles import AbstractUserRole


class Teacher(AbstractUserRole):
    available_permissions = {
        "edit_patient_file": True,
    }
