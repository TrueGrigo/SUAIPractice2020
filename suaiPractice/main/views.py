from django.shortcuts import render
import numpy as np
import matplotlib.pyplot as plt
import os
from django.conf import settings
import io
import requests
from  django.http import HttpResponse
from PIL import Image
# Create your views here.


def index(request):
    n = 11
    nx, ny = 50, 50
    #  x = map([59.9783333333333,59.8883333333333,59.8850000000000, 59.9166666666667, 59.9083333333333,59.9900000000000 , 59.9583333333333,59.9366666666667,60.9383333333333,59.9383333333333])
    x = [59.9783333333333, 59.8883333333333, 59.8850000000000, 59.9166666666667, 59.9083333333333, 59.9900000000000,
         59.9583333333333, 59.9366666666667, 60, 59.9383333333333, 59.9016666666667]
    y = [30.2166666666667, 30.2183333333333, 30.1666666666667, 30.2633333333333, 29.9366666666667, 29.8583333333333,
         29.7966666666667, 29.7883333333333, 29.9383333333333, 30.1283333333333, 30.0866666666667]
    z = [2.04999995200000, 9.60000038100000, 11.2500000000000, 2.60444400000000, 13.8000001900000, 13.8999996200000,
         15.6999998100000, 6.30999994300000, 12.5000000000000, 15.4200000800000, 35.2000007600000]
    n = len(x)
    nx = 10*n
    ny = 10*n
    xi = np.linspace(min(x), max(x), nx)
    yi = np.linspace(min(y), max(y), ny)
    xi, yi = np.meshgrid(xi, yi)
    xi, yi = xi.flatten(), yi.flatten()
    maxx=max(x)
    minx=min(x)
    maxy=max(y)
    miny=min(y)

    grid3 = linear_rbf(x, y, z, xi, yi)
    grid3 = grid3.reshape((ny, nx))
    # Comparisons...
    # plt.title("Scipy's Rbf with function=linear")

    plot(x, y, z, grid3)
    plt.savefig("/var/www/u0981570/data/www/grigothedeveloper.ru.com/suaiPractice/main/static/filename.png", bbox_inches = 'tight',pad_inches = 0)
    im = Image.open("/var/www/u0981570/data/www/grigothedeveloper.ru.com/suaiPractice/main/static/filename.png")
    out=im
    #out = im.transpose(Image.ROTATE_90)
    #out.putalpha(128)
    prextileX=round(XTiles(minx,11))
    prextileY=round(YTiles(miny,11))
    maxYtile=round(YTiles(maxy,11))
    maxXtile=round(XTiles(maxx,11))

    xtiless=-maxXtile+prextileX+1
    ytiless=maxYtile-prextileY+1
    xlen = xtiless*256
    ylen = ytiless*256
    out = out.resize((xlen, ylen))
    dxt=maxx-minx
    dxy=maxy-miny
    h, w= out.size
    dx=dxt/w
    dy=dxy/h

    iteri=0
    xtileList=[]
    xtileList.append(prextileX)
    ytileList=[]
    ytileList.append(prextileY)
    prevx=0
    prevy=0
    for curx in range (0,w,1):
        pox=minx+curx*dx
        if prextileX!=round(XTiles(pox,11)):
            for cury in range (0,h,1):
                poy=miny+cury*dy
                if prextileY!=round(YTiles(poy,11)):
                    prextileY=round(YTiles(poy,11))
                    ytileList.append(prextileY)
                    croppred=out.crop((curx,cury,curx+256,cury+256))
                   # pth='/var/www/u0981570/data/www/grigothedeveloper.ru.com/suaiPractice/main/static/11/tile-test-'+str(xtileList[-1])+'-'+str(ytileList[-1])+'.png'
                  # croppred.save(pth, 'png')
            prextileX=round(XTiles(pox,11))
            xtileList.append(prextileX)
    tileslistX=[]
    tileslistY = []
    for xe in range(n):
        for ye in range(n):
            if round(XTiles(x[xe],11)) in tileslistX and round(YTiles(y[ye],11)) in tileslistY:
                print("")
            else:
                tileslistY.append(round(YTiles(y[ye],11)))
                tileslistX.append(round(XTiles(x[xe],11)))

    xlen = len(set(tileslistY))*256
    ylen = len(set(tileslistX))*256
    out = out.resize((xlen, ylen))
    dw=-256
    dh =-256
    h, w= out.size
    dx=dxt/w
    dy=dxy/h
    xtileList2=[]
    ytileList2=[]
    xw=0
    itery=0
    iterx=-1
    for poy in range(0,ylen,256):
        for pod in range(0,xlen,256 ):
            iterx=iterx+1
            xtileList2.append(XTiles(minx+iterx*dx,11))
            chas=out.crop((pod,poy,pod+256,poy+256))
            ytileList2.append(ytileList[iterx-1])
            pth='/var/www/u0981570/data/www/grigothedeveloper.ru.com/suaiPractice/main/static/11/tile-'+str(xtileList[itery-1])+'-'+str(ytileList[iterx-1])+'.png'
            response = requests.get('http://vec02.maps.yandex.net/tiles?l=map&x='+str(ytileList[iterx-1])+'&y='+str(xtileList[itery-1])+'&z=11&g=Ga')
            ocmtiles = Image.open(io.BytesIO(response.content))
            ocmtiles = ocmtiles.convert("RGBA")
            datas = ocmtiles.getdata()
            newData = []
            for item in datas:
                if item[0] == 185 and item[1] == 223 and item[2] == 245:
                    newData.append((0, 0, 0, 0))
                else:
                    newData.append(item)

            ocmtiles.putdata(newData)
            chas.paste(ocmtiles, (0, 0), ocmtiles)
            chas.save(pth, 'png')
        iterx=0
        itery=itery+1
       
    return render(request, 'main/index.html',{'xtiles':xtiless,'ytiles':ytiless,'ytileList':ytileList,'xtileList':xtileList,'ytileList2':ytileList2,'xtileList2':xtileList2,'prextileX':prextileX,'curcs':curx,'width':w,'iteri':iteri,'pop':pox ,'hight':h,'dxt':dxt,'dxy':dxy,'dx':dx,'dy':dy})
def linear_rbf(x, y, z, xi, yi):
    dist = distance_matrix(x,y, xi,yi)

    # Mutual pariwise distances between observations
    internal_dist = distance_matrix(x,y, x,y)

    # Now solve for the weights such that mistfit at the observations is minimized
    weights = np.linalg.solve(internal_dist, z)

    # Multiply the weights for each interpolated point by the distances
    zi =  np.dot(dist.T, weights)
    return zi

def distance_matrix(x0, y0, x1, y1):
    obs = np.vstack((x0, y0)).T
    interp = np.vstack((x1, y1)).T

    # (Yay for ufuncs!)
    d0 = np.subtract.outer(obs[:,0], interp[:,0])
    d1 = np.subtract.outer(obs[:,1], interp[:,1])

    return np.hypot(d0, d1)

def plot(x,y,z,grid):
    plt.figure()
    plt.imshow(grid, extent=(min(x), max(x), max(y), min(y)))
    #plt.axes().set_aspect(1./plt.axes().get_data_ratio())
    plt.gca().set_axis_off()
    plt.margins(0, 0)
    plt.scatter(x,y,c=z)

def XTiles(lat, z):
    lat = lat * np.math.pi / 180.0

    a = 6378137
    k = 0.0818191908426
    z1 = np.math.tan(np.math.pi / 4 + lat / 2) / pow(np.math.tan(np.math.pi / 4 + np.math.asin(k * np.math.sin(lat)) / 2), k)
    pix_Y = round((20037508.342789 - a * np.math.log(z1)) * 53.5865938 / pow(2, 23 - z))
    return (pix_Y / 256)


def YTiles(lat, z):
    lon = lat * np.math.pi / 180.0
    a = 6378137
    k = 0.0818191908426
    pix_X = round((20037508.342789 + a * lon) * 53.5865938 / pow(2.0, 23 - z))
    return (pix_X / 256)

