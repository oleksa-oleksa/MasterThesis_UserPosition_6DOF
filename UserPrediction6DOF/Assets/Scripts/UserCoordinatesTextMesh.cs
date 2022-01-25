using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using System;
using System.Text;
using System.IO;


public class UserCoordinatesTextMesh : MonoBehaviour
{
    public GameObject textDisplay;
    public TextMeshPro textmeshPro;

    // Camera 
    private Vector3 headposition;
    private Quaternion orientation;
    private Vector3 orientationEulerAngles;
    private Vector3 velocity;
    private double speed;

    // time variables for delay
    private float waitTime = 1.0f;
    private float timer = 0.0f;


    void Start()
    {
      
    }

  
    void Update()
    {
        
        timer += Time.deltaTime;
                
        float x = Camera2CSV.headposition.x;
        float y = Camera2CSV.headposition.y;
        float z = Camera2CSV.headposition.z;

        /*
        float qx = Camera2CSV.orientation.x;
        float qy = Camera2CSV.orientation.y;
        float qz = Camera2CSV.orientation.z;
        float qw = Camera2CSV.orientation.w;

        float speed = (float)Camera2CSV.speed;

        */

        float eu_x_pitch = Camera2CSV.orientationEulerAngles.x; // transverse axis 
        float eu_y_yaw = Camera2CSV.orientationEulerAngles.y; // vertical axis
        float eu_z_roll = Camera2CSV.orientationEulerAngles.z; // longitudial axis

        // Check if we have reached the time limit
        if (timer > waitTime)
        {
            // update text block in hologram 
            textmeshPro = GetComponent<TextMeshPro>();
            textmeshPro.SetText("Position x: {0:3}, y: {1:3}, z: {2:3}; \r\n " +
                "Pitch x: {3:3}, Yaw y: {4:3}, Roll z: {5:3}",
               x, y, z, eu_x_pitch, eu_y_yaw, eu_z_roll);

            // reset timer
            timer = 0.0f;

        }
    }


}
