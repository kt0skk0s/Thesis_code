using UnityEngine;

public class AvatarController : MonoBehaviour
{
    private SpriteRenderer spriteRenderer;

    void Awake()
    {
        spriteRenderer = GetComponent<SpriteRenderer>();
    }

    public void SetEmotion(string emotion)
    {
        if (emotion == "Positive")
            spriteRenderer.color = Color.green;

        else if (emotion == "Negative")
            spriteRenderer.color = Color.red;

        else if (emotion == "Neutral")
            spriteRenderer.color = Color.blue;
    }
}