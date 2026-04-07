from huggingface_hub import login
import getpass

print("="*60)
print("Hugging Face CLI Login")
print("="*60)
print("\nTo get your token:")
print("1. Go to https://huggingface.co/settings/tokens")
print("2. Create a new token with 'Write' permission")
print("3. Copy and paste it below\n")

token = getpass.getpass("Enter your Hugging Face token: ")

try:
    login(token=token, add_to_git_credential=True)
    print("\n✅ Successfully logged in to Hugging Face!")
    print("Your credentials are saved and ready for deployment.")
except Exception as e:
    print(f"\n❌ Login failed: {e}")
