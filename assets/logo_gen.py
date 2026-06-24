"""Generate the turboswarm logo: a topographic contour map of a multimodal
optimization landscape (warm rings = peak, cool rings = valley) with the swarm
converging on the glowing global optimum. Transparent (alpha) background.

Writes assets/logo.svg. To regenerate the raster assets afterwards:

    python assets/logo_gen.py
    python -c "import cairosvg; \\
        cairosvg.svg2png(url='assets/logo.svg', write_to='assets/logo-512.png', output_width=512, output_height=512); \\
        cairosvg.svg2png(url='assets/logo.svg', write_to='assets/logo-256.png', output_width=256, output_height=256); \\
        cairosvg.svg2png(url='assets/logo.svg', write_to='docs/assets/favicon.png', output_width=256, output_height=256)"

Run from the repository root. Requires `pip install cairosvg` for rasterizing.
"""
import math

def gauss(x, y, cx, cy, s, a):
    return a * math.exp(-((x-cx)**2 + (y-cy)**2) / (2*s*s))

def f(x, y):
    z  = gauss(x, y, -0.35, -0.85, 0.52,  1.00)   # tall peak (top)
    z += gauss(x, y,  0.70, -0.55, 0.48,  0.52)   # medium peak (top-right)
    z += gauss(x, y,  0.05,  0.55, 0.50, -1.05)   # deep valley = optimum (bottom)
    z += gauss(x, y, -0.72,  0.35, 0.50, -0.30)   # shallow dip
    return z

RANGE = 1.95
G = 230
def lerp(a,b,k): return a+(b-a)*k
def ramp(t, stops):
    t=max(0.0,min(1.0,t))
    for i in range(len(stops)-1):
        t0,c0=stops[i]; t1,c1=stops[i+1]
        if t<=t1:
            k=(t-t0)/(t1-t0) if t1>t0 else 0
            return tuple(int(lerp(c0[j],c1[j],k)) for j in range(3))
    return stops[-1][1]

JET = [(0.0,(0x1e,0x40,0xaf)),(0.30,(0x06,0xb6,0xd4)),(0.52,(0x22,0xc5,0x5e)),
       (0.74,(0xf5,0x9e,0x0b)),(1.0,(0xef,0x44,0x44))]

grid=[[f((gx/(G-1)*2-1)*RANGE,(gy/(G-1)*2-1)*RANGE) for gy in range(G)] for gx in range(G)]
ZMIN=min(min(r) for r in grid); ZMAX=max(max(r) for r in grid)

# Raw screen coords (top-down), squashed vertically for subtle perspective
SQ=0.84
def rx(gx): return gx/(G-1)*440
def ry(gy): return gy/(G-1)*440*SQ

# Levels: closed rings around peak (positive) and valley (negative), skip the ~0 plane
peak_fracs   = [0.20,0.36,0.54,0.73,0.92]
valley_fracs = [0.16,0.30,0.46,0.63,0.80,0.95]
levels=[("peak",fr,ZMAX*fr) for fr in peak_fracs]+[("val",fr,ZMIN*fr) for fr in valley_fracs]

def interp(p,q,vp,vq,lv):
    t=(lv-vp)/(vq-vp) if vq!=vp else 0.5
    return p+(q-p)*t

segs=[]  # (color, width, x1,y1,x2,y2)
for kind,fr,lv in levels:
    t=(lv-ZMIN)/(ZMAX-ZMIN)
    col=ramp(t,JET)
    width = lerp(2.0, 3.6, fr)            # inner rings (high frac) bolder
    chex=f"#{col[0]:02x}{col[1]:02x}{col[2]:02x}"
    for gx in range(G-1):
        for gy in range(G-1):
            v00=grid[gx][gy]; v10=grid[gx+1][gy]; v11=grid[gx+1][gy+1]; v01=grid[gx][gy+1]
            idx=(1 if v00>lv else 0)|(2 if v10>lv else 0)|(4 if v11>lv else 0)|(8 if v01>lv else 0)
            if idx in (0,15): continue
            x0,x1=rx(gx),rx(gx+1); y0,y1=ry(gy),ry(gy+1)
            eb=(interp(x0,x1,v00,v10,lv),y0)
            er=(x1,interp(y0,y1,v10,v11,lv))
            et=(interp(x0,x1,v01,v11,lv),y1)
            el=(x0,interp(y0,y1,v00,v01,lv))
            table={1:[(el,eb)],2:[(eb,er)],3:[(el,er)],4:[(er,et)],6:[(eb,et)],
                   7:[(el,et)],8:[(et,el)],9:[(et,eb)],11:[(et,er)],12:[(er,el)],
                   13:[(er,eb)],14:[(eb,el)],5:[(el,et),(eb,er)],10:[(el,eb),(et,er)]}
            for pa,pb in table.get(idx,[]):
                segs.append((chex,width,pa[0],pa[1],pb[0],pb[1]))

# optimum at the deep valley
ow,oh=0.05,0.55
oX=(ow/RANGE+1)/2*440
oY=(oh/RANGE+1)/2*440*SQ

# swarm spiralling into optimum
parts=[]
NP=11
for i in range(NP):
    ang=i*2.39996; rad=12+62*(i/(NP-1))
    px=oX+math.cos(ang)*rad; py=oY+math.sin(ang)*rad*SQ
    rr=2.8+5.2*(1-i/(NP-1)); op=0.5+0.5*(1-i/(NP-1))
    parts.append((px,py,rr,op))

# auto-fit (include rings, swarm, halo)
xs=[]; ys=[]
for c,w,x1,y1,x2,y2 in segs: xs+=[x1,x2]; ys+=[y1,y2]
HALO=34
xs+=[oX-HALO,oX+HALO]; ys+=[oY-HALO,oY+HALO]
for px,py,rr,op in parts: xs+=[px-rr,px+rr]; ys+=[py-rr,py+rr]
minx,maxx=min(xs),max(xs); miny,maxy=min(ys),max(ys)
M=34
s=min((512-2*M)/(maxx-minx),(512-2*M)/(maxy-miny))
tx=(512-(maxx-minx)*s)/2-minx*s
ty=(512-(maxy-miny)*s)/2-miny*s

lines="".join(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{c}" stroke-width="{w:.2f}" stroke-linecap="round"/>' for c,w,x1,y1,x2,y2 in segs)
sw=""
for px,py,rr,op in parts:
    sw+=f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{oX:.1f}" y2="{oY:.1f}" stroke="#5566a3" stroke-width="1.1" stroke-opacity="{op*0.18:.2f}" stroke-linecap="round"/>'
    sw+=f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{rr:.1f}" fill="#5566a3" fill-opacity="{op:.2f}"/>'

svg=f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">
<defs><radialGradient id="opt" cx="0.5" cy="0.5" r="0.5">
<stop offset="0" stop-color="#06b6d4" stop-opacity="0.6"/>
<stop offset="1" stop-color="#06b6d4" stop-opacity="0"/></radialGradient></defs>
<g transform="translate({tx:.1f},{ty:.1f}) scale({s:.4f})">
{lines}
<circle cx="{oX:.1f}" cy="{oY:.1f}" r="34" fill="url(#opt)"/>
{sw}
<circle cx="{oX:.1f}" cy="{oY:.1f}" r="10" fill="#0891b2"/>
<circle cx="{oX:.1f}" cy="{oY:.1f}" r="4.5" fill="#ffffff"/>
</g></svg>'''

open("assets/logo.svg","w").write(svg)
print(f"levels={len(levels)} segs={len(segs)} fit s={s:.3f}")
