# Add any constraints or validation functions here
def validate_email(email):
    return len(email) > 0 and "@" in email
