using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using System;
using System.Text;
using System.IO;
using Unity.Profiling.LowLevel.Unsafe;


public class timestamp_formats : MonoBehaviour
{
    public GameObject textDisplay;
    public TextMeshPro textmeshPro;

    // time variables for delay
    private float waitTime = 1.0f;
    private float timer = 0.0f;


    void Start()
    {

    }


    void Update()
    {

        timer += Time.deltaTime;

        float timestamp = Time.time;
        //float timestamp_ms = Time.time * 1000.0f;
        //int timestamp_dt = DateTime.UtcNow.Millisecond;
        long timestamp_nano = ProfilerUnsafeUtility.Timestamp * ProfilerUnsafeUtility.TimestampToNanosecondsConversionRatio.Numerator / ProfilerUnsafeUtility.TimestampToNanosecondsConversionRatio.Denominator;

        if (timer > waitTime)
        {
            // update text block in hologram 
            textmeshPro = GetComponent<TextMeshPro>();
            textmeshPro.SetText("Time.time: {0:17}, Time nano: {1:17}", timestamp, timestamp_nano);

            // reset timer
            timer = 0.0f;

        }
    }


}
