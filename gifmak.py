from PIL import Image, ImageDraw,ImageFont
import numpy as np

def createIFirst(x=300):

    im = Image.new("RGB", (x, x))
    font = ImageFont.truetype("arial.ttf", 30)#ImageFont.load("arial.pil")
    cen=divide9(x,x)
    draw = ImageDraw.Draw(im)
    draw.text((100,100),'dos IAs \n redes neuronales\n(mal hechas XD) \n jugando gato entre ellas' , font=font, fill=(255,255,255))
    return im.copy()
def createIm(mat,x=300):

    im = Image.new("RGB", (x, x))
    font = ImageFont.truetype("arial.ttf", 30)#ImageFont.load("arial.pil")
    cen=divide9(x,x)
    draw = ImageDraw.Draw(im)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            t=''
            if mat[i,j]==-.5:
                t='X'
            elif mat[i,j]==.5:
                t='O'
            draw.text((cen[i],cen[j]),t , font=font, fill=(255,255,255))
    return im.copy()

def divide9(x,y):
    a=int(x/3)
    b=int(y/3)
    r=[i*a for i in range(4)]
    cen=[]
    for i in range(len(r)-1):
        tmp=r[i]+r[i+1]
        tmp=tmp/2
        cen.append(tmp)

    #print(r)
    #print(cen)
    return cen
def creategif(name,frames,x=450):
    arr=[]
    arr.append(createIFirst(x=x))
    arr.append(createIFirst(x=x))
    arr.append(createIFirst(x=x))
    for frame in frames:
        arr.append(createIm(frame,x=x))
    arr[0].save(name+'.gif',
                   save_all=True,
                   append_images=arr[1:],
                   duration=350,
                   loop=0)

if __name__ == '__main__':
    divide9(200,200)
