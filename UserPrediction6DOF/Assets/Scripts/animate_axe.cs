using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class animate_axe : MonoBehaviour
{
    public GameObject animationObject_axe;
    private float secondsPerFrame = 0.044f;

    private int currentFrame = 0;
    private float timeAccumulator;

    protected GameObject getFrame(int n)
    {
        return animationObject_axe.transform.GetChild(n).gameObject;
    }

    protected int getFrameCount()
    {
        return animationObject_axe.transform.childCount;
    }

    void Start()
    {
        animationObject_axe = GameObject.Find("Animation_Axe");
        timeAccumulator = 0;

        for (int i = 0; i < getFrameCount(); i++)
        {
            getFrame(i).SetActive(false);
        }
    }

    // Update is called once per frame
    void Update()
    {

        timeAccumulator += Time.deltaTime;

        if (timeAccumulator > secondsPerFrame)
        {
            getFrame(currentFrame).SetActive(false);
            currentFrame = (currentFrame + Mathf.RoundToInt(timeAccumulator / secondsPerFrame)) % getFrameCount();
            getFrame(currentFrame).SetActive(true);
            timeAccumulator = 0.0f;
        }
    }

}
