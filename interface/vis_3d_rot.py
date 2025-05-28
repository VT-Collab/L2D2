import os, sys
import pygame
from OpenGL.GL import *

from pygame.locals import *
from pygame.constants import *
from OpenGL.GLU import *

import socket
import pickle

class OBJ:
    generate_on_init = True
    @classmethod
    def loadTexture(cls, imagefile):
        surf = pygame.image.load(imagefile)
        image = pygame.image.tostring(surf, 'RGBA', 1)
        ix, iy = surf.get_rect().size
        texid = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texid)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
        return texid

    @classmethod
    def loadMaterial(cls, filename):
        contents = {}
        mtl = None
        dirname = os.path.dirname(filename)

        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'newmtl':
                mtl = contents[values[1]] = {}
            elif mtl is None:
                raise ValueError("mtl file doesn't start with newmtl stmt")
            elif values[0] == 'map_Kd':
                # load the texture referred to by this declaration
                mtl[values[0]] = values[1]
                imagefile = os.path.join(dirname, mtl['map_Kd'])
                mtl['texture_Kd'] = cls.loadTexture(imagefile)
            else:
                mtl[values[0]] = list(map(float, values[1:]))
        return contents

    def __init__(self, filename, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        self.gl_list = 0
        dirname = os.path.dirname(filename)

        material = None
        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(list(map(float, values[1:3])))
            elif values[0] in ('usemtl', 'usemat'):
                material = values[1]
            elif values[0] == 'mtllib':
                self.mtl = self.loadMaterial(os.path.join(dirname, values[1]))
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, norms, texcoords, material))
        if self.generate_on_init:
            self.generate()

    def generate(self):
        self.gl_list = glGenLists(1)
        glNewList(self.gl_list, GL_COMPILE)
        glEnable(GL_TEXTURE_2D)
        glFrontFace(GL_CCW)
        for face in self.faces:
            vertices, normals, texture_coords, material = face

            mtl = self.mtl[material]
            if 'texture_Kd' in mtl:
                # use diffuse texmap
                glBindTexture(GL_TEXTURE_2D, mtl['texture_Kd'])
            else:
                # just use diffuse colour
                glColor(*mtl['Kd'])

            glBegin(GL_POLYGON)
            for i in range(len(vertices)):
                if normals[i] > 0:
                    glNormal3fv(self.normals[normals[i] - 1])
                if texture_coords[i] > 0:
                    glTexCoord2fv(self.texcoords[texture_coords[i] - 1])
                glVertex3fv(self.vertices[vertices[i] - 1])
            glEnd()
        glDisable(GL_TEXTURE_2D)
        glEndList()

    def render(self):
        glCallList(self.gl_list)

    def free(self):
        glDeleteLists([self.gl_list])



def normalize_obj(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        vertices = []
        min_coords = [float('inf')] * 3
        max_coords = [float('-inf')] * 3

        for line in infile:
            if line.startswith('v'):
                x, y, z = map(float, line.split()[1:])
                vertices.append([x, y, z])
                min_coords = [min(min_coords[i], c) for i, c in enumerate([x, y, z])]
                max_coords = [max(max_coords[i], c) for i, c in enumerate([x, y, z])]

        range_coords = [max_coords[i] - min_coords[i] for i in range(3)]

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if line.startswith('v'):
                x, y, z = map(float, line.split()[1:])
                normalized_coords = [(c - min_coords[i]) / range_coords[i] for i, c in enumerate([x, y, z])]
                outfile.write(f'v {normalized_coords[0]} {normalized_coords[1]} {normalized_coords[2]}\n')
            else:
                outfile.write(line)


def visualize_3d_gripper():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('localhost', 8080)
    s.connect(server_address)
    
    os.environ['SDL_VIDEO_WINDOW_POS'] = "-1500, 500"  
    pygame.init()
    display = (640, 480)
    screen = pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    
    glMatrixMode(GL_PROJECTION) # <---- specify projection matrix
    gluPerspective(45, (640/480), 0.1, 500)

    glMatrixMode(GL_MODELVIEW)  # <---- specify model view matrix
    modelMatrix = glGetFloatv(GL_MODELVIEW_MATRIX)
    # import file
    model = OBJ('normalized_gripper.obj')

    rx, ry, rz = (0.,0.,0.)
    init = False

    while True:
        glPushMatrix()
        glLoadIdentity()
        if not init:
            glTranslatef(0.5, 0.5, 0.5)
            glRotate(180, 1, 0, 0)
            # glRotate(10, 0, 1, 0)
        #     glRotate(0, 0, 0, 1)
            glTranslatef(-0.5, -0.5, -0.5)
            init = True

        data = s.recv(4096)
        rx, ry, rz = pickle.loads(data)

        for e in pygame.event.get():
            if e.type == QUIT:
                sys.exit()

        glTranslatef(0.5, 0.5, 0.5)
        glRotate(rx, 1, 0, 0)
        glRotate(ry, 0, 1, 0)
        glRotate(rz, 0, 0, 1)
        glTranslatef(-0.5, -0.5, -0.5)

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glMultMatrixf(modelMatrix)
        modelMatrix = glGetFloatv(GL_MODELVIEW_MATRIX)
        glLoadIdentity()
        glTranslatef(-0.5, -0.5, -3.)
        glMultMatrixf(modelMatrix)

        model.render()
        glPopMatrix()

        pygame.display.flip()
        pygame.time.wait(1)


if __name__ == "__main__":
#     input_file = 'gripper.obj'
#     output_file = 'normalized_gripper.obj'
#     normalize_obj(input_file, output_file)
    visualize_3d_gripper()