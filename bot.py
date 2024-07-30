import numpy as np
import pygame
from moviepy.editor import ImageSequenceClip
import telebot
import os
import time
from requests.exceptions import ConnectionError, Timeout

# Inserisci il tuo token qui
API_TOKEN = 'YOUR TELEGRAM TOKEN'
bot = telebot.TeleBot(API_TOKEN)

# Funzione per creare uno stato iniziale casuale
def make_state():
    d = np.random.uniform(0.3, 1.0)
    vx = np.random.uniform(0.3, 2.0)
    vy = np.random.uniform(0.3, 2.0)
    middle_state = [0, 0, vx, vy]
    left = [d, 0, -middle_state[2]/2, -middle_state[3]/2]
    right = [-d, 0, -middle_state[2]/2, -middle_state[3]/2]
    return np.array(middle_state + left + right)

# Definizione della classe Body
class Body:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = np.array(position)
        self.velocity = np.array(velocity)

    def compute_acceleration(self, other_bodies, G=1.0, softening=0.2):
        acceleration = np.zeros(2)
        for other in other_bodies:
            if other is not self:
                r = other.position - self.position
                distance = np.linalg.norm(r)
                acceleration += G * other.mass * r / np.maximum(distance**3, softening**3)
        return acceleration

    def draw(self, frame, system, positions, win):
        scale = 500
        x = positions[frame, system.bodies.index(self), 0] * scale + 500
        y = 1000 - (positions[frame, system.bodies.index(self), 1] * scale + 500)
        pygame.draw.circle(win, (225,225,225), (int(x), int(y)), 3)
        if frame > 2:
            updatedPoints = []
            for point in positions[:frame, system.bodies.index(self)]:
                x = point[0] * scale + 500
                y = 1000 - (point[1] * scale + 500)
                updatedPoints.append((int(x), int(y)))
            pygame.draw.lines(win, (255,255,255), False, updatedPoints, 1)

    def get_state(self):
        state = np.array([self.position[0], self.position[1], self.velocity[0], self.velocity[1]])
        return state

# Definizione della classe System
class System:
    def __init__(self, G=1.0, state=None, bodies=None):
        self.G = G
        if state is not None:
            self.bodies = []
            for body in range(int(len(state)/4)):
                n = int(body*4)
                self.bodies.append(Body(mass=1.0, position=[state[0+n], state[1+n]], velocity=[state[2+n], state[3+n]]))
        else:
            self.bodies = bodies

    def compute_accelerations(self):
        accelerations = []
        for body in self.bodies:
            other_bodies = [b for b in self.bodies if b is not body]
            accelerations.append(body.compute_acceleration(other_bodies, self.G))
        return accelerations

    def compute_total_energy(self):
        kinetic_energy = 0.5 * sum(body.mass * np.dot(body.velocity, body.velocity) for body in self.bodies)

        potential_energy = 0
        for i, body1 in enumerate(self.bodies):
            for body2 in self.bodies[i+1:]:
                distance = np.linalg.norm(body2.position - body1.position)
                potential_energy -= self.G * body1.mass * body2.mass / distance

        total_energy = kinetic_energy + potential_energy
        return total_energy

    def get_state(self):
        state = []
        for body in self.bodies:
            body_state = body.get_state()
            state.extend(body_state)
        return np.array(state)

    def integrate(self, dt, num_steps, save_positions=False):
        if save_positions:
            positions = np.zeros((num_steps, len(self.bodies), 2))
            delta_energy = np.zeros(num_steps)
            initial_energy = self.compute_total_energy()

        for step in range(num_steps):
            accelerations = self.compute_accelerations()

            for i, body in enumerate(self.bodies):
                body.position += body.velocity * dt + 0.5 * accelerations[i] * dt**2

            new_accelerations = self.compute_accelerations()

            for i, body in enumerate(self.bodies):
                body.velocity += 0.5 * (accelerations[i] + new_accelerations[i]) * dt

            if save_positions:
                positions[step] = [body.position for body in self.bodies]
                delta_energy[step] = initial_energy - self.compute_total_energy()

        if save_positions: 
            return positions, delta_energy
        else:
            return self.get_state()

# Funzione per generare il video della simulazione
def create_simulation_video():
    initial_state = make_state()
    system = System(state=initial_state)
    num_steps = 900  # 15 secondi * 60 fps
    dt = 0.01
    positions, delta_energy = system.integrate(dt, num_steps, save_positions=True)

    pygame.init()
    size = (1000, 1000)
    screen = pygame.display.set_mode(size)  # Imposta la modalità display
    clock = pygame.time.Clock()
    
    # Path to save video
    video_path = "simulation.mp4"
    frames = []

    for i in range(positions.shape[0]):
        screen.fill((10,10,10))
        for body in system.bodies:
            body.draw(i, system, positions, screen)

        pygame.display.flip()
        pygame.image.save(screen, f"frame_{i:04d}.png")
        frames.append(f"frame_{i:04d}.png")
        clock.tick(60)  # Frame rate

    pygame.quit()

    # Create video from frames
    clip = ImageSequenceClip(frames, fps=60)
    clip.write_videofile(video_path, codec='libx264')

    # Clean up frame images
    for frame in frames:
        os.remove(frame)

    return video_path

# Funzione del comando /simulation per il bot Telegram
@bot.message_handler(commands=['simulation'])
def simulation_command(message):
    bot.send_message(message.chat.id, "Attendere 15 secondi, simulazione in corso...")

    while True:
        try:
            video_path = create_simulation_video()
            with open(video_path, 'rb') as video_file:
                bot.send_video(message.chat.id, video_file, caption="Here is your simulation video!")
            break  # Esci dal loop se il video è stato inviato con successo
        except (ConnectionError, Timeout) as e:
            print(f"Errore di connessione o timeout: {e}. Riprovo tra 10 secondi...")
            time.sleep(10)  # Attendere 10 secondi prima di riprovare
        except Exception as e:
            print(f"Errore imprevisto: {e}. Riprovo tra 10 secondi...")
            time.sleep(10)  # Attendere 10 secondi prima di riprovare

# Main function to start the bot
def main():
    bot.polling(none_stop=True)

if __name__ == '__main__':
    main()
