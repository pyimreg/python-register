class SalientPoint
    {
	public:
		int X;
        int Y; 
        float Saliency;

        SalientPoint()
		{
		}


        SalientPoint(int x, int y, float saliency)
        {
            X = x;
            Y = y;
            Saliency = saliency;
        }

    };