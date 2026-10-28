using UnityEngine;
using UnityEngine.InputSystem;

public class EmotionManager : MonoBehaviour
{
    public AvatarController avatarController;

    void Update()
    {
        if (Keyboard.current == null)
            return;

        if (Keyboard.current.digit1Key.wasPressedThisFrame)
        {
            SetEmotion("Positive");
        }

        if (Keyboard.current.digit2Key.wasPressedThisFrame)
        {
            SetEmotion("Negative");
        }

        if (Keyboard.current.digit3Key.wasPressedThisFrame)
        {
            SetEmotion("Neutral");
        }
    }

    public void SetEmotion(string emotion)
    {
        if (avatarController != null)
        {
            avatarController.SetEmotion(emotion);
        }
    }
}