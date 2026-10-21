# Bunya paso a paso (lo que funcionó)

Registro de la instalación desde cero con la cuenta de staff `uqcvaras`, comando por comando.
Solo se anota aquí lo que se probó en Bunya y funcionó. Sin scripts y sin variables.

## 1. Entrar

Desde PowerShell en la laptop. Pide la contraseña UQ y luego el push de Okta Verify.

```powershell
ssh uqcvaras@bunya.rcc.uq.edu.au
```

## 2. Clonar el repo (rama develop)

Si ya existe `~/computer-vision-tracking` de un intento anterior, borrarlo primero.

```bash
rm -rf ~/computer-vision-tracking
git clone --depth 1 -b develop https://github.com/CateHey/computer-vision-tracking.git ~/computer-vision-tracking
cd ~/computer-vision-tracking
git log --oneline -1
```

Actualizar después de cada cambio subido desde la laptop:

```bash
cd ~/computer-vision-tracking && git pull
```

## 3. Subir el video

Desde PowerShell en la laptop, no desde Bunya. Ejemplo con el clip de 10 s del repo:

```powershell
scp C:\Users\uqcvaras\Downloads\computer-vision-tracking\data\clips\output-10s.mp4 uqcvaras@bunya.rcc.uq.edu.au:computer-vision-tracking/data/raw/original.mp4
```

Comprobar en Bunya:

```bash
ls -lh ~/computer-vision-tracking/data/raw/original.mp4
```

## 4. Cargar conda

`module load miniforge` sin versión falla con "exist but cannot be loaded as requested".
Hay que dar la versión completa. Ver las disponibles con `module spider miniforge`.

```bash
module load miniforge/26.3.2-1
```

(en curso)

## Pendiente: cuenta Slurm

Sin cuenta Slurm, `sbatch`, `salloc` y `srun` fallan con
`Invalid account or account/partition combination specified`.
`sacctmgr -n show assoc user=uqcvaras format=account%30` sale vacío: la cuenta de staff
no tiene ninguna asociación. Hay que pedir a un admin del grupo (lista en el email de
aprobación de RCC) que agregue `uqcvaras` a la cuenta Slurm del grupo. La cuenta de
estudiante anterior (`s4948012`) sí tenía una por defecto, por eso antes funcionaba
sin `--account`.
